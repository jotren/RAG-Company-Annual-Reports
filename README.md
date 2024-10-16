# Company Annual Report Retrieval-Augmented Generation 

## Project Overview

The aim of this project is to explore and implement Retrieval-Augmented Generation (RAG) in various applications. RAG is typically broken down into a few key steps:

- **Query**: The user submits a query to the system.
- **Retrieval**: Based on the query, the system returns the text chunks it deems most relevant.
- **Generation**: The returned chunks, along with the query, are sent to a generative model (such as GPT) to provide an answer based on the retrieved results.

The retrieval component of this process is usually implemented using a vector database, where the text is transformed into compact vector representations that summarize the semantic space.

## Data

To create an initial Minimum Viable Product (MVP), I used company report PDFs from some of the largest public companies in the UK and US. The selected companies include:

- Unilever
- Apple
- Amazon
- Manchester United

These publicly available, dense PDFs are well-suited for implementing Retrieval-Augmented Generation.

## LLM Model

The model used for vectorization is the [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). This model is specifically designed for generating sentence embeddings, providing fixed-size vector representations that capture the semantic meaning of sentences. This allows for effective comparisons and similarity searches. Being relatively small in size, it can be hosted on a local machine.

## Approach

The approach for this project involves using FAISS for vector generation and GPT-3.5 for generating responses. FAISS (Facebook AI Similarity Search) is a highly efficient library for vector searches:

```python
def process_pdfs_and_store_faiss(pdf_folder, pdf_files, tokenizer, model, faiss_index, text_chunk_df, doc_chunks):
    """Process all PDFs, tokenize, vectorize, and store embeddings in FAISS."""
    doc_ids = []  # Track which document corresponds to which vectors
    all_embeddings = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Processing: {pdf_file}")
        
        # Step 1: Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Step 2: Chunk text into smaller segments based on sentences and a 512 token limit
        chunks = chunk_text(text, tokenizer, max_tokens=512)
        
        # Step 3: Generate embeddings for each chunk
        embeddings = model.encode(chunks, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()  # Convert to numpy for FAISS
        
        # Step 4: Add embeddings to FAISS index
        faiss_index.add(embeddings)
        
        # Keep track of document IDs and embedding count for reference
        doc_ids.extend([pdf_file] * len(embeddings))
        all_embeddings.append(embeddings)
        
        # Append text chunks and file name to the DataFrame for FAISS and final retrieval
        for chunk in chunks:
            row_to_add = pd.DataFrame({"chunk": [chunk], "file": [pdf_file]})
            text_chunk_df = pd.concat([text_chunk_df, row_to_add], ignore_index=True)
        
        # BM25 setup: Tokenizing chunks for keyword search
        for chunk in chunks:
            doc_chunks.append(pdf_file)  # Track which document the chunk belongs to
    
    return doc_ids, all_embeddings, text_chunk_df, doc_chunks

```
This function processes each PDF, converts the content into strings, and generates FAISS vector representations for the text chunks.

Next, we query this database by converting the query into vectors and performing either:

- __Euclidean Distance Calculations__ : Measures the direct distance between Euclidean vectors for the query and sample texts, returning the smallest distance.
- __Cosine Similarity Calculations__: Evaluates the angle difference from the origin in the semantic space between the query and sample texts, returning the smallest angle.

Both methods yield results, but cosine similarity is generally preferred for text similarity scoring. The comparison between both methods in my functions showed no clear winner.

```python

def faiss_only_retrieval_cosine(query, faiss_index, model, text_chunk_df, doc_chunks, top_n=5):
    """FAISS retrieval returning text chunk, file, and similarity score using cosine similarity."""
    
    # Step 1: Use Sentence-BERT to embed the query
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    
    # Normalize the query embedding for cosine similarity
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Step 2: Use FAISS to find the top_n most similar documents to the query embedding
    D, I = faiss_index.search(query_embedding, k=top_n)  # Note: D contains inner products now

    results = []
    for idx in range(len(I[0])):  # Iterate only over valid indices returned by FAISS
        if I[0][idx] < len(text_chunk_df):  # Ensure the index is within bounds
            chunk_idx = I[0][idx]
            chunk_text = text_chunk_df.iloc[chunk_idx]["chunk"]
            file_name = text_chunk_df.iloc[chunk_idx]["file"]
            # Use inner product as the similarity score
            similarity_score = D[0][idx]  # D contains inner products now
            results.append({
                "file_name": file_name,
                "text_chunk": chunk_text,
                "similarity_score": similarity_score
            })
    
    return results

def faiss_only_retrieval_euclidean(query, faiss_index, model, text_chunk_df, doc_chunks, top_n=5):
    """FAISS retrieval returning text chunk, file, and similarity score."""
    
    # Step 1: Use Sentence-BERT to embed the query
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    
    # Step 2: Use FAISS to find the top_n most similar documents to the query embedding
    D, I = faiss_index.search(query_embedding, k=top_n)

    results = []
    for idx in range(len(I[0])):  # Iterate only over valid indices returned by FAISS
        if I[0][idx] < len(text_chunk_df):  # Ensure the index is within bounds
            chunk_idx = I[0][idx]
            chunk_text = text_chunk_df.iloc[chunk_idx]["chunk"]
            file_name = text_chunk_df.iloc[chunk_idx]["file"]
            similarity_score = 1 / (1 + D[0][idx])  # Convert distance to similarity score
            results.append({
                "file_name": file_name,
                "text_chunk": chunk_text,
                "similarity_score": similarity_score
            })
    
    return results

```
## Result

To minimise costs, I have not integrated a GPT model into this project; however, I have explored its capabilities in other projects.

- [GPT Shopping Assistant](https://github.com/jotren/LLM-GPT-Shopping-Assistant)
- [GPT Database Integration](https://github.com/jotren/GPT-Database-Integration)

I conducted tests to evaluate the effectiveness of the retrieval process. Below are examples of results for both a broad and a concise query.

### Broad Query

```python
query = "What was the revenue for Amazon in 2022"
# results = faiss_only_retrieval_cosine(query, index, model, text_chunk_df)
results = faiss_only_retrieval_euclidean(query, index, model, text_chunk_df)

# Print the results
for result in results:
    print(f"File: {result['file_name']}, Similarity: {result['similarity_score']}")
    print(f"Text Chunk: {result['text_chunk']}\n")
```
Top Result:

``` 
Year Ended December 31,
2021 2022 Net Sales: North America $ 279,833  $ 315,880  International 127,787  118,007  AWS 62,202  80,096 Consolidated$ 469,822 $ 513,983 
```

### Concise Query

```python
query = "What is happening to the electric car market? Is it expected to grow or shrink?"
# results = faiss_only_retrieval_cosine(query, index, model, text_chunk_df)
results = faiss_only_retrieval_euclidean(query, index, model, text_chunk_df)

# Print the results
for result in results:
    print(f"File: {result['file_name']}, Similarity: {result['similarity_score']}")
    print(f"Text Chunk: {result['text_chunk']}\n")

```
Top Result: 

```
 As a result, the market for our vehicles could be 
negatively affected by numerous factors, such as: 
•
perceptions about electric vehicle features, quality, safety, performance and cost;
•
perceptions about the limited range over which electric vehicles may be driven on a single battery charge, and access to charging 
facilities; 
•
competition, including from other types of alternative fuel vehicles, plug-in hybrid electric vehicles and high fuel-economy internal 
combustion engine vehicles; 
•
volatility in the cost of oil, gasoline and energy, such as wide fluctuations in crude oil prices during 2020; 
•
government regulations and economic incentives and conditions; and
•
concerns about our future viability.
```

You can see in both cases the model returned the most relevant text.

### Key Takeaways

- The technology is currently limited by the token limit of the GPT model. This limitation is expected to change as token limits increase.
- Initially I used BM25 to try and improve the retrieval. This algorithmn is only useful when the text chunks vary enormously with size.
- If you do chunk text, try and keep the paragraphs together, this is to retain semantic meaning in chunks.
- 