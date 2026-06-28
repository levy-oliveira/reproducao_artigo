Passos para execução com Ollama:

1. Baixar Ollama: https://ollama.com
2. Após instalação, baixar o modelo Qwen3:8B: 'ollama pull qwen3:8b'
3. Ver se está instalado: 'ollama list'
4. Testa o modelo no CMD: 'ollama run qwen3:8b' , depois digita Hello e se tiver dado certo o resultado, Ctrl + D
5. Baixar as coisas em requirements.txt: 'pip install -r requirements.txt'
6. Passar para data/jdoctor o arquivo que você vai fazer o teste.
7. Mudar em src\config.py o nome do json que irá ser lido.
8. Rodar o experimento com 'python src/main.py'
9. O output será salvo em outputs.
10. Mudar o nome do output para: predictions_##Tag_##_##, trocando ## pela respectiva Tag, valor e SR ou noSR, com base no Json de entrada.