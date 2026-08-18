# ============================================================
# SMART PDF / DOCUMENT KNOWLEDGE EXTRACTOR
# Review 2 Prototype  (hardened)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONSTANTS / GUARDRAILS
# ============================================================

MAX_FILES = 20
MAX_FILE_SIZE_MB = 30
MIN_WORDS_FOR_KEYWORDS = 5  # below this, TF-IDF on the corpus is unreliable


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart PDF Knowledge Extractor",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Smart PDF / Document Knowledge Extractor")

st.write(
    """
    Upload multiple PDF documents and the system will:

    • Extract text from each PDF
    • Identify important keywords (via corpus-wide TF-IDF)
    • Calculate similarity between documents
    • Show a similarity matrix
    • Visualize relationships between documents
    """
)


# ============================================================
# PDF TEXT EXTRACTION (cached, per-file error isolation)
# ============================================================

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes, filename):
    """
    Extract text from all pages of a PDF.
    Returns (text, error_message). error_message is None on success.
    Handles corrupted files and encrypted files (tries an empty password).
    """
    try:
        reader = PdfReader(file_bytes)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return "", "File is password-protected and could not be opened."

        full_text = ""
        for page in reader.pages:
            try:
                text = page.extract_text()
            except Exception:
                text = None
            if text:
                full_text += text + "\n"

        return full_text, None

    except PdfReadError:
        return "", "File is not a valid or readable PDF (corrupted?)."
    except Exception as e:
        return "", f"Unexpected error while reading file: {e}"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.warning(
        f"You uploaded {len(uploaded_files)} files. Only the first "
        f"{MAX_FILES} will be processed to keep things responsive."
    )
    uploaded_files = uploaded_files[:MAX_FILES]


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_files:

    documents = []
    document_names = []
    skipped = []  # (filename, reason)

    # --------------------------------------------------------
    # EXTRACT TEXT FROM EVERY PDF (isolated failures)
    # --------------------------------------------------------

    progress = st.progress(0.0, text="Extracting text from PDFs...")

    for i, pdf_file in enumerate(uploaded_files):

        size_mb = pdf_file.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            skipped.append(
                (pdf_file.name, f"File too large ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB limit).")
            )
            progress.progress((i + 1) / len(uploaded_files))
            continue

        file_bytes = pdf_file.read()
        text, error = extract_text_from_pdf(file_bytes, pdf_file.name)

        if error:
            skipped.append((pdf_file.name, error))
        elif not text.strip():
            skipped.append(
                (pdf_file.name, "No extractable text found (likely a scanned/image-only PDF).")
            )
        else:
            documents.append(clean_text(text))
            document_names.append(pdf_file.name)

        progress.progress((i + 1) / len(uploaded_files))

    progress.empty()

    if documents:
        st.success(f"{len(documents)} PDF file(s) processed successfully.")

    if skipped:
        with st.expander(f"⚠️ {len(skipped)} file(s) skipped — click for details"):
            for name, reason in skipped:
                st.write(f"**{name}**: {reason}")

    # --------------------------------------------------------
    # SHOW EXTRACTION RESULTS
    # --------------------------------------------------------

    if documents:

        st.header("📄 Document Information")

        information = [
            {
                "Document": document_names[i],
                "Characters": len(documents[i]),
                "Words": len(documents[i].split()),
            }
            for i in range(len(documents))
        ]

        df_information = pd.DataFrame(information)
        st.dataframe(df_information, use_container_width=True)

        # ----------------------------------------------------
        # TF-IDF VECTORIZATION (computed once, reused for
        # both keywords and similarity — no duplicate work,
        # and keywords are now meaningful since IDF is
        # computed across the whole corpus, not a single doc)
        # ----------------------------------------------------

        st.header("🧠 Document Vectorization")

        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()

            st.write(
                f"Documents converted into {tfidf_matrix.shape[1]} TF-IDF features."
            )

            # ------------------------------------------------
            # KEYWORDS (derived from the corpus-wide matrix)
            # ------------------------------------------------

            st.header("🔑 Extracted Keywords")

            dense = tfidf_matrix.toarray()

            for i in range(len(documents)):

                if len(documents[i].split()) < MIN_WORDS_FOR_KEYWORDS:
                    st.subheader(document_names[i])
                    st.caption("Document too short for reliable keyword extraction.")
                    continue

                row = dense[i]
                top_indices = row.argsort()[::-1][:10]
                keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]

                st.subheader(document_names[i])
                st.write(" • ".join(keywords) if keywords else "No distinctive keywords found.")

            # ------------------------------------------------
            # COSINE SIMILARITY
            # ------------------------------------------------

            similarity_matrix = cosine_similarity(tfidf_matrix)

            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=document_names,
                columns=document_names,
            )

            st.header("🔗 Document Similarity Matrix")
            st.dataframe(similarity_df.round(2), use_container_width=True)

            st.download_button(
                "⬇️ Download similarity matrix as CSV",
                data=similarity_df.round(4).to_csv().encode("utf-8"),
                file_name="similarity_matrix.csv",
                mime="text/csv",
            )

            # ------------------------------------------------
            # MOST SIMILAR PAIR
            # ------------------------------------------------

            if len(document_names) >= 2:

                best_score = -1
                best_pair = ("", "")

                for i in range(len(document_names)):
                    for j in range(i + 1, len(document_names)):
                        score = similarity_matrix[i][j]
                        if score > best_score:
                            best_score = score
                            best_pair = (document_names[i], document_names[j])

                st.subheader("🏆 Most Similar Documents")
                st.write(f"**{best_pair[0]}** and **{best_pair[1]}**")
                st.write(f"Similarity Score: **{best_score:.2f}**")
            else:
                st.info("Upload at least 2 valid PDFs to compare documents and see a similarity graph.")

            # ------------------------------------------------
            # SIMILARITY GRAPH
            # ------------------------------------------------

            if len(document_names) >= 2:

                st.header("🕸️ Document Relationship Graph")

                threshold = st.slider(
                    "Minimum similarity to draw a connection",
                    min_value=0.0, max_value=1.0, value=0.10, step=0.05
                )

                graph = nx.Graph()
                for document in document_names:
                    graph.add_node(document)

                for i in range(len(document_names)):
                    for j in range(i + 1, len(document_names)):
                        similarity = similarity_matrix[i][j]
                        if similarity >= threshold:
                            graph.add_edge(
                                document_names[i], document_names[j], weight=similarity
                            )

                fig, ax = plt.subplots(figsize=(10, 7))

                if len(graph.nodes) > 0:
                    positions = nx.spring_layout(graph, seed=42)

                    # size nodes by how connected they are, so hubs stand out
                    degrees = dict(graph.degree())
                    node_sizes = [1800 + 500 * degrees[n] for n in graph.nodes]

                    nx.draw_networkx_nodes(
                        graph, positions, node_size=node_sizes,
                        node_color="#6ea8fe", ax=ax
                    )
                    nx.draw_networkx_labels(graph, positions, font_size=9, ax=ax)
                    nx.draw_networkx_edges(graph, positions, width=2, ax=ax, edge_color="#999999")

                    edge_labels = nx.get_edge_attributes(graph, "weight")
                    edge_labels = {edge: f"{value:.2f}" for edge, value in edge_labels.items()}
                    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, ax=ax)

                ax.set_title("Relationship Between Uploaded Documents")
                ax.axis("off")

                st.pyplot(fig)
                plt.close(fig)  # avoid leaking figures across reruns

        except ValueError:
            st.error(
                "The uploaded documents do not contain enough distinct, "
                "extractable text for comparison (e.g. all documents are "
                "near-empty or contain only stop words)."
            )

    else:
        st.error("None of the uploaded files could be processed. See the details above.")

else:

    st.info("👆 Upload two or more PDF documents to begin.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart PDF / Document Knowledge Extractor | "
  
)