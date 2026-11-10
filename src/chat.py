from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Digite 'sair' para encerrar.")

    while True:
        try:
            question = input("\nVocê: ").strip()

            if question.lower() in ("sair", "exit", "quit"):
                print("Chat encerrado.")
                break

            if not question:
                continue

            answer = chain.invoke(question)

            print(f"\nAssistente: {answer}")

        except (KeyboardInterrupt, EOFError):
            print("\nChat encerrado.")
            break

        except Exception as error:
            print(f"Erro ao processar a pergunta: {error}")


if __name__ == "__main__":
    main()
