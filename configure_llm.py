import yaml, os, openai
from llama_index.llms.openai import OpenAI
from huggingface_hub import HfApi, HfFolder
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
                                ServiceContext, 
                                set_global_service_context, 
                                SimpleDirectoryReader
                                )
from constants import *

os.chdir(working_dir)

with open('cadentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

openai.api_key = credentials['OPENAI_API_KEY']
os.environ["OPENAI_API_KEY"] = credentials['OPENAI_API_KEY']
os.environ["LLAMA_CLOUD_API_KEY"] = credentials['LLAMA_CLOUD_API_KEY']

HfApi(token=credentials['HUGGINGFACEHUB_API_TOKEN'])
folder = HfFolder()
folder.save_token(credentials['HUGGINGFACEHUB_API_TOKEN'])

llm_model = credentials['OPENAI_GPT4_ENGINE'] if gpt_flag == 'GPT4' else credentials['OPENAI_GPT3_ENGINE']

if embedding_flag == 'OPENAI':
    embedding_llm = OpenAIEmbedding(
                                model="text-embedding-3-small",
                                api_key=credentials['OPENAI_API_KEY']
                                )
    
else:
    embedding_llm = HuggingFaceEmbedding(
                                        model_name="avsolatorio/GIST-Embedding-v0",
                                        device='mps'
                                        )

chat_llm = OpenAI(
                api_key=credentials['OPENAI_API_KEY'],
                model=llm_model,
                temperature=0.3
                )

creative_llm = OpenAI(
                    api_key=credentials['OPENAI_API_KEY'],
                    model=llm_model,
                    temperature=0.4
                    )

completion_llm = OpenAI(
                        api_key=credentials['OPENAI_API_KEY'],
                        model=llm_model,
                        temperature=0
                        )

service_context = ServiceContext.from_defaults(
                                            embed_model=embedding_llm,
                                            chunk_size=2000,
                                            llm=chat_llm
                                            )

set_global_service_context(service_context)