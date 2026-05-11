from io import BytesIO

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from pdf_reader_core import (
    assess_extraction_quality,
    build_extraction_diagnostics,
    build_structured_summary_prompt,
    check_ocr_availability,
    ensure_text_was_extracted,
    flatten_structured_pages_with_ids,
    load_settings,
    merge_page_texts,
    ocr_pdf_bytes,
)
from pdf_structure import extract_structured_pdf


MODEL_CONTEXT_SIZES = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
}


def init_page():
    st.set_page_config(page_title="Ask My PDF(s)", page_icon="PDF")
    st.sidebar.title("Nav")
    st.session_state.setdefault("costs", [])
    st.session_state.setdefault("document_history", [])


def select_model():
    settings = load_settings()
    if settings.llm_provider == "ollama":
        st.sidebar.markdown(f"Local LLM: `{settings.ollama_llm_model}`")
        st.session_state.model_name = settings.ollama_llm_model
        st.session_state.max_token = 0
        return build_llm(settings)

    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    else:
        st.session_state.model_name = "gpt-4"

    st.session_state.max_token = MODEL_CONTEXT_SIZES[st.session_state.model_name] - 300
    return build_llm(settings, model_name=st.session_state.model_name)


def build_llm(settings, model_name=None):
    if settings.llm_provider == "ollama":
        return OllamaLLM(model=settings.ollama_llm_model, base_url=settings.ollama_base_url)
    return ChatOpenAI(temperature=0, model_name=model_name or "gpt-3.5-turbo", api_key=settings.openai_api_key)


def get_pdf_text():
    uploaded_file = st.file_uploader(label="Upload your PDF here", type="pdf")
    if not uploaded_file:
        return None

    try:
        pages = extract_structured_pdf(uploaded_file)
        text = merge_page_texts([page.raw_text for page in pages])
        text = ensure_text_was_extracted(text)
    except Exception as exc:
        st.error(str(exc))
        return None

    settings = load_settings()
    if settings.embedding_provider == "openai":
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=settings.embedding_model_name,
            chunk_size=500,
            chunk_overlap=0,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return text_splitter.split_text(text)


def extract_structured_pdf_with_optional_ocr(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    pages = extract_structured_pdf(BytesIO(pdf_bytes))
    input_quality = assess_extraction_quality(pages)
    settings = load_settings()
    ocr_result = {
        "applied": False,
        "input_quality": input_quality.to_dict(),
        "output_quality": None,
        "availability": check_ocr_availability(settings).to_dict(),
        "error": None,
    }

    if input_quality.needs_ocr:
        availability = check_ocr_availability(settings)
        ocr_result["availability"] = availability.to_dict()
        if availability.available:
            st.info("Extracted text is sparse. Running OCR before re-extracting the PDF.")
            try:
                ocr_bytes = ocr_pdf_bytes(pdf_bytes, settings)
                ocr_pages = extract_structured_pdf(BytesIO(ocr_bytes))
                output_quality = assess_extraction_quality(ocr_pages)
                ocr_result.update(
                    {
                        "applied": True,
                        "output_quality": output_quality.to_dict(),
                    }
                )
                pages = ocr_pages
            except Exception as exc:
                ocr_result["error"] = str(exc)
                st.warning(f"OCR failed, continuing with the original extraction: {exc}")
        elif settings.ocr_enabled:
            message = availability.message
            ocr_result["error"] = message
            st.warning(message)

    st.session_state["last_ocr_result"] = ocr_result

    text = merge_page_texts([page.raw_text for page in pages])
    ensure_text_was_extracted(text)
    return pages


def get_structured_pdf():
    uploaded_file = st.file_uploader(label="Upload your PDF here", type="pdf")
    if not uploaded_file:
        return None

    try:
        return extract_structured_pdf_with_optional_ocr(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        return None


def build_embeddings():
    settings = load_settings()
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)
    return OpenAIEmbeddings(model=settings.embedding_model_name, api_key=settings.openai_api_key)


def load_qdrant():
    settings = load_settings()
    client = QdrantClient(path=settings.qdrant_path)

    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if settings.collection_name not in collection_names:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=settings.vector_size, distance=Distance.COSINE),
        )
        print("collection created")

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=build_embeddings(),
    )


def get_qdrant_collection_status():
    settings = load_settings()
    client = QdrantClient(path=settings.qdrant_path)
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        exists = settings.collection_name in collection_names
        count = client.count(collection_name=settings.collection_name, exact=True).count if exists else 0
        return {
            "qdrant_path": settings.qdrant_path,
            "collection_name": settings.collection_name,
            "exists": exists,
            "record_count": count,
        }
    finally:
        client.close()


def get_ocr_status():
    settings = load_settings()
    availability = check_ocr_availability(settings)
    return {
        "ocr_enabled": settings.ocr_enabled,
        "ocr_command": settings.ocr_command,
        "ocr_language": settings.ocr_language,
        "availability": availability.to_dict(),
    }


def delete_qdrant_collection():
    settings = load_settings()
    client = QdrantClient(path=settings.qdrant_path)
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if settings.collection_name not in collection_names:
            return False
        client.delete_collection(collection_name=settings.collection_name)
        return True
    finally:
        client.close()


def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    try:
        qdrant.add_texts(pdf_text)
    finally:
        qdrant.client.close()


def build_vector_store_with_metadata(texts, metadatas, ids=None):
    qdrant = load_qdrant()
    try:
        qdrant.add_texts(texts, metadatas=metadatas, ids=ids)
    finally:
        qdrant.client.close()


def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )


def summarize_structured_pages(llm, pages, user_instruction=None):
    response = llm.invoke(build_structured_summary_prompt(pages, user_instruction=user_instruction))
    return getattr(response, "content", response)


def close_qa_model(qa):
    vectorstore = getattr(getattr(qa, "retriever", None), "vectorstore", None)
    client = getattr(vectorstore, "client", None)
    if client:
        client.close()


def format_sources(answer):
    if not isinstance(answer, dict):
        return []

    sources = []
    seen = set()
    for document in answer.get("source_documents", []):
        metadata = getattr(document, "metadata", {}) or {}
        key = (
            metadata.get("document_id"),
            metadata.get("page"),
            metadata.get("section_type"),
            metadata.get("heading"),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "page": metadata.get("page"),
                "section_type": metadata.get("section_type"),
                "heading": metadata.get("heading"),
                "keywords": metadata.get("keywords"),
                "document_type": metadata.get("document_type"),
                "needs_ocr": metadata.get("needs_ocr"),
                "document_id": metadata.get("document_id"),
            }
        )

    return sources


def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        pages = get_structured_pdf()
        if pages:
            st.session_state.structured_pages = pages
            extraction_quality = assess_extraction_quality(pages)
            diagnostics = build_extraction_diagnostics(pages)
            last_ocr_result = st.session_state.get("last_ocr_result")
            document_type = diagnostics["document_type"]["document_type"]
            page_summaries = [
                {
                    "page": page.page_number,
                    "characters": len(page.raw_text),
                    "sections": len(page.sections),
                    "tables": len(page.tables),
                    "keywords": page.keywords,
                }
                for page in pages
            ]
            if extraction_quality.needs_ocr:
                st.warning("Extracted text is sparse. This PDF may need OCR before reliable search or summaries.")
            st.markdown("## Structured pages")
            st.json(page_summaries)
            st.markdown("## Extraction diagnostics")
            st.json(diagnostics)
            if last_ocr_result:
                st.markdown("## OCR result")
                st.json(last_ocr_result)

            texts, metadatas, ids = flatten_structured_pages_with_ids(pages)
            with st.spinner("Loading PDF ..."):
                try:
                    build_vector_store_with_metadata(texts, metadatas, ids=ids)
                    st.session_state.document_history.append(
                        {
                            "document_id": metadatas[0].get("document_id") if metadatas else None,
                            "document_type": document_type,
                            "pages": len(pages),
                            "records": len(texts),
                            "needs_ocr": extraction_quality.needs_ocr,
                            "ocr_applied": last_ocr_result.get("applied") if last_ocr_result else False,
                        }
                    )
                    st.success(f"Structured PDF data was saved to the vector database. Records: {len(texts)}")
                except Exception as exc:
                    st.error(f"Failed to save PDF text to Qdrant or OpenAI embeddings: {exc}")


def page_vector_db_admin():
    st.title("Vector DB Admin")
    try:
        st.markdown("## Collection status")
        st.json(get_qdrant_collection_status())
    except Exception as exc:
        st.error(f"Failed to read Qdrant status: {exc}")

    try:
        st.markdown("## OCR status")
        st.json(get_ocr_status())
    except Exception as exc:
        st.error(f"Failed to read OCR status: {exc}")

    st.markdown("## Uploaded document history")
    history = st.session_state.get("document_history", [])
    if history:
        st.json(history)
    else:
        st.info("No documents have been uploaded in this Streamlit session.")

    st.markdown("## Danger zone")
    confirm_delete = st.checkbox("I understand this deletes the configured Qdrant collection.")
    if st.button("Delete configured collection", disabled=not confirm_delete):
        try:
            deleted = delete_qdrant_collection()
            if deleted:
                st.success("Configured Qdrant collection was deleted.")
            else:
                st.info("Configured Qdrant collection did not exist.")
        except Exception as exc:
            st.error(f"Failed to delete Qdrant collection: {exc}")


def ask(qa, query):
    try:
        if load_settings().llm_provider == "ollama":
            return qa.invoke({"query": query}), 0.0

        with get_openai_callback() as cb:
            answer = qa.invoke({"query": query})

        return answer, cb.total_cost
    finally:
        close_qa_model(qa)


def page_ask_my_pdf():
    st.title("Ask My PDF(s)")

    llm = select_model()
    answer_mode = st.radio("Answer mode", ["Retrieval QA", "Structured summary"])
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        elif answer_mode == "Structured summary":
            pages = st.session_state.get("structured_pages")
            if not pages:
                st.warning("Upload a PDF first so structured page data is available.")
                answer = None
            else:
                with st.spinner("Summarizing structured PDF data ..."):
                    answer = summarize_structured_pages(llm, pages, user_instruction=query)
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                if isinstance(answer, dict):
                    st.write(answer.get("result", answer))
                    sources = format_sources(answer)
                    if sources:
                        st.markdown("## Sources")
                        st.json(sources)
                else:
                    st.write(str(answer))


def main():
    init_page()

    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)", "Vector DB Admin"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()
    elif selection == "Vector DB Admin":
        page_vector_db_admin()

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == "__main__":
    main()
