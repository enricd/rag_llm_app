import os
import streamlit as st
from openai import AzureOpenAI

# Definindo variáveis de ambiente
endpoint = os.getenv("ENDPOINT_URL", "https://ragteste.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo-16k")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "6d546bb39e4443ad908ebb704f356cdc")

# Inicializando o cliente Azure OpenAI com autenticação baseada em chave
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# Função para obter resposta da API do Azure OpenAI
def get_completion(user_input):
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um economista do Observatório FIESC que entende muito de economia. As suas especialidades são: dados de emprego, dados de comércio exterior"
                },
                {
                    "role": "user",
                    "content": user_input  # Usar a entrada do usuário aqui
                },
                  {  "role": "assistant",
                    "content": "Prezado, a quantidade de empregos em Santa Catarina em 2024 é de 230 mil empregos"}
            ],
            max_tokens=200,
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        # Acessando a resposta de forma correta
        response_content = completion.choices[0].message.content
        return response_content
    except Exception as e:
        return f"Erro ao obter resposta: {e}"

# Interface com Streamlit
st.title("Desabafe com a zel")

user_input = st.text_input("Digite seu lamento:")

if st.button("Enviar"):
    if user_input:
        response = get_completion(user_input)
        st.write(response)
    else:
        st.write("Por favor, insira uma pergunta.")
