from langchain_huggingface import HuggingFaceEndpointEmbeddings,ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

#INDEXING STEPS 1(A)
video_id="PnqJllk3RfA"
yt_api=YouTubeTranscriptApi()
try:
 transcript_list= yt_api.fetch(video_id=video_id,languages=["en"])

 transcript=""
 for segment in transcript_list:
  transcript+= segment.text + " "


except TranscriptsDisabled:
 print("No transcripts availiable for this video")
print(transcript)

#SPLITTER 1(B)
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks= splitter.create_documents([transcript])
print(len(chunks))

#VECTOR STORE 1(C) & EMBEDDINGS 1(D)

embeddings= HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vector_store=FAISS.from_documents(chunks,embeddings)


##RETRIEVAL STEP-2

retreiver=vector_store.as_retriever(search_type="similarity",kwargs={"k":4})


#AUGMENTATION STEP-3
prompt= PromptTemplate(template="""you are a helpful assistant.
Answer only from the provided transcript context.
If the context is insufficient just say you don't know
context:{context}
Question:{question}""",
input_variables=['context','question'])

#creating the context

question="Was the topic how to make world happier discussed in the podcast ?"
retirved_docs=retreiver.invoke(question)

context_text =""
for docs in retirved_docs:
 context_text+= docs.page_content+ " "


final_prompt= prompt.invoke({'question':question,'context':context_text})
print(final_prompt)

#GENERATION STEP-4
llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",task="text-generation")
model=ChatHuggingFace(llm=llm)

answer=model.invoke(final_prompt)
print(answer.content)