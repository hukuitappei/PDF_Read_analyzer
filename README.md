# PDF Reader

StreamlitとLangChainを使用したPDFリーダーアプリケーションです。

## 機能

- PDFファイルのアップロード
- PDFテキストの抽出とベクトル化
- Qdrantを使用したベクトルデータベースの構築
- OpenAI Embeddingsを使用したテキスト埋め込み

## セットアップ

### 1. 環境の構築

```bash
# uvでPython 3.10環境を作成（既に完了済み）
uv init --python 3.10

# 依存関係をインストール（既に完了済み）
uv sync
```

### 2. 環境変数の設定

1. `env.example` を `.env` にコピー
2. `.env` ファイルを開いて、OpenAI API キーを設定

```bash
copy env.example .env
```

`.env` ファイルの内容例：
```
OPENAI_API_KEY=your_actual_openai_api_key_here
EMBEDDING_MODEL_NAME=text-embedding-ada-002
QDRANT_PATH=./local_qdrant
COLLECTION_NAME=my_collection
```

### 3. アプリケーションの実行

Windows環境では：
```bash
run.bat
```

または直接：
```bash
uv run streamlit run pdf_uploader.py
```

## 使用方法

1. アプリケーションを起動すると、ブラウザでStreamlitアプリが開きます
2. サイドバーから「PDF Upload」を選択
3. PDFファイルをアップロード
4. アップロードされたPDFは自動的にベクトルデータベースに保存されます

## 技術スタック

- **Python**: 3.10
- **Streamlit**: Webアプリケーションフレームワーク
- **LangChain**: LLMアプリケーション開発フレームワーク
- **PyPDF2**: PDFファイル処理
- **Qdrant**: ベクトルデータベース
- **OpenAI**: テキスト埋め込み
- **uv**: Pythonパッケージ管理

## ファイル構成

```
pdf_reader/
├── pdf_uploader.py      # メインアプリケーション
├── pyproject.toml       # プロジェクト設定と依存関係
├── env.example          # 環境変数テンプレート
├── run.bat             # Windows実行スクリプト
├── README.md           # このファイル
└── local_qdrant/       # Qdrantデータベース（自動生成）
```

## 注意事項

- OpenAI API キーが必要です
- 初回実行時にQdrantデータベースが自動的に作成されます
- PDFファイルのサイズや複雑さによって処理時間が変わります
