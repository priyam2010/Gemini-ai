from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # Use a suitable model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# 1. Load Data
# Replace 'path/to/your/report.pdf' with your file
loader = PyPDFLoader(r"C:\Users\bhatt\Downloads\report\Oil-India-Notification-2025.pdf")
docs = loader.load()

# 2. Chunk Documents (Optional but recommended for very long docs)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, # Adjust based on model context limit, leaving room for prompts
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(docs)

print(f"Original documents: {len(docs)}, Split documents: {len(split_docs)}")

# Optional: If your document is very long, you may use split_docs
# If the report is shorter and fits, you can use the original `docs` or a single `Stuff` chain.
# 3. Create Summarization Chain
# Use 'map_reduce' for long documents. 'refine' is an alternative for highly contextual docs.
map_reduce_chain = load_summarize_chain(
    llm, 
    chain_type="map_reduce",
    verbose=True # Set to True to see the steps
)

# 4. Invoke the Chain and Generate Summary
final_summary = map_reduce_chain.invoke(split_docs) # Use split_docs for MapReduce

# 5. Condense to Target Length (Post-processing)
# The initial summary might still be long. Use a final prompt to condense it to 2 pages.
condense_prompt = f"""
Condense the following comprehensive summary into a 2-page executive summary,
focusing on key campaign performance metrics, major achievements, and critical challenges:

SUMMARY: {final_summary['output_text']}

EXECUTIVE SUMMARY (2 Pages Max):
"""

executive_summary = llm.invoke(condense_prompt).content
print("\n--- Executive Summary (2 Pages) ---\n", executive_summary)
# Example dummy comments (in a real scenario, load from a file)
customer_comments = [
    "The new feature is amazing, 5 stars! The setup was so easy.",
    "I had to wait 30 minutes for support, completely unacceptable.",
    "The product works fine, nothing special, but it's reliable.",
    "I love the new color options, great job to the design team!",
    # ... 496 more comments
]

# Join into a single text for LLM processing
full_feedback_text = "\n---\n".join(customer_comments)
sentiment_prompt_template = """
Analyze the following customer comments. Your goal is to condense all 500 comments 
into key sentiment highlights (Positive, Negative, Neutral).

Output a structured summary containing:
1. Overall Sentiment Split (e.g., 60% Positive, 30% Negative, 10% Neutral).
2. 3-5 Key Positive Highlights/Themes (with a few example phrases).
3. 3-5 Key Negative Highlights/Themes (with a few example phrases).
4. 1-2 Key Neutral Observations.

CUSTOMER COMMENTS:
{comments}

KEY SENTIMENT HIGHLIGHTS:
"""

sentiment_prompt = sentiment_prompt_template.format(comments=full_feedback_text)
# You can use a direct call for this, as the full feedback is condensed into the prompt.
# If the full text of 500 comments exceeds the context window, you would need to 
# use the MapReduce chain again, where the 'Map' step analyzes sentiment for smaller batches, 
# and the 'Reduce' step aggregates the findings.

sentiment_summary = llm.invoke(sentiment_prompt).content
print("\n--- Key Sentiment Highlights ---\n", sentiment_summary)