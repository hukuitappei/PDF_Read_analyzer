# PDF Read Analyzer (TypeScript)

TypeScript + Express + LangChain.js + OpenAI で PDF 質問応答アプリを動かすプロジェクトです。

## 機能

- PDF ファイルのアップロード
- テキスト抽出とチャンク分割
- OpenAI Embeddings でベクトル化
- 質問に対して PDF 文脈ベースで回答

## セットアップ

1. 依存関係をインストール

```bash
npm install
```

2. 環境変数を設定

```bash
copy env.example .env
```

`.env` に `OPENAI_API_KEY` を設定してください。

## 実行

開発モード:

```bash
npm run dev
```

ブラウザで `http://localhost:3000` を開いて利用します。

本番ビルド:

```bash
npm run build
npm start
```

## API

- `POST /api/upload` (`multipart/form-data`, field: `pdf`)
- `POST /api/ask` (`application/json`, body: `{ "question": "..." }`)

## 補足

- ベクトルDBは `MemoryVectorStore` を使用しているため、サーバー再起動でインデックスは消えます。
- 永続化したい場合はQdrantなどの外部ベクトルDBへ差し替えてください。
