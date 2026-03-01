import "dotenv/config";
import path from "node:path";

import express, { Request, Response } from "express";
import multer from "multer";
import OpenAI from "openai";
import pdfParse from "pdf-parse";

type Chunk = {
  id: number;
  filename: string;
  text: string;
  embedding: number[];
};

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(express.json({ limit: "5mb" }));
app.use(express.static(path.resolve("public")));

const port = Number(process.env.PORT ?? "3000");
const embeddingModel = process.env.EMBEDDING_MODEL_NAME ?? "text-embedding-3-small";
const chatModel = process.env.CHAT_MODEL_NAME ?? "gpt-4o-mini";

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is required in .env");
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
let indexedChunks: Chunk[] = [];

function splitText(text: string, chunkSize = 500, overlap = 50): string[] {
  const cleaned = text.replace(/\r/g, "").trim();
  if (!cleaned) return [];

  const chunks: string[] = [];
  let start = 0;

  while (start < cleaned.length) {
    const end = Math.min(start + chunkSize, cleaned.length);
    chunks.push(cleaned.slice(start, end));
    if (end === cleaned.length) break;
    start = Math.max(end - overlap, start + 1);
  }

  return chunks;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    aNorm += a[i] * a[i];
    bNorm += b[i] * b[i];
  }

  if (!aNorm || !bNorm) return 0;
  return dot / (Math.sqrt(aNorm) * Math.sqrt(bNorm));
}

async function embedTexts(texts: string[]): Promise<number[][]> {
  const response = await openai.embeddings.create({
    model: embeddingModel,
    input: texts,
  });

  return response.data.map((item) => item.embedding);
}

app.post("/api/upload", upload.single("pdf"), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: "PDF file is required" });
      return;
    }

    const parsed = await pdfParse(req.file.buffer);
    const chunks = splitText(parsed.text ?? "");

    if (chunks.length === 0) {
      res.status(400).json({ error: "No readable text found in PDF" });
      return;
    }

    const embeddings = await embedTexts(chunks);
    indexedChunks = chunks.map((text, index) => ({
      id: index + 1,
      filename: req.file!.originalname,
      text,
      embedding: embeddings[index],
    }));

    res.json({
      message: "PDF indexed successfully",
      filename: req.file.originalname,
      chunks: indexedChunks.length,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to process PDF" });
  }
});

app.post("/api/ask", async (req: Request, res: Response) => {
  try {
    const question = String(req.body?.question ?? "").trim();
    if (!question) {
      res.status(400).json({ error: "question is required" });
      return;
    }

    if (indexedChunks.length === 0) {
      res.status(400).json({ error: "Upload a PDF first" });
      return;
    }

    const [questionEmbedding] = await embedTexts([question]);
    const scored = indexedChunks
      .map((chunk) => ({
        ...chunk,
        score: cosineSimilarity(questionEmbedding, chunk.embedding),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);

    const context = scored
      .map((chunk, idx) => `[${idx + 1}] ${chunk.text}`)
      .join("\n\n");

    const completion = await openai.chat.completions.create({
      model: chatModel,
      temperature: 0,
      messages: [
        {
          role: "system",
          content:
            "You answer only from provided context. If context is insufficient, say you do not know.",
        },
        {
          role: "user",
          content: `Question: ${question}\n\nContext:\n${context}`,
        },
      ],
    });

    res.json({
      answer: completion.choices[0]?.message?.content ?? "No answer generated",
      sources: scored.map((chunk) => ({
        id: chunk.id,
        filename: chunk.filename,
        score: Number(chunk.score.toFixed(4)),
      })),
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to answer question" });
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
