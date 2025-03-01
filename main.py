from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG!")
    print(app.get_graph().draw_ascii()) # pip install grandalf
    print(app.get_graph().draw_mermaid())
    print(app.invoke(input={"question": "what is agent memory?"}))
