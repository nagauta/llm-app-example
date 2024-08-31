import { promises as fs } from "fs";
import { Document, HuggingFaceEmbedding, Ollama, Settings, VectorStoreIndex } from "llamaindex";

console.log(`go...`);
Settings.llm = new Ollama({
    model: "llama3.1",
  });

  Settings.embedModel = new HuggingFaceEmbedding({
    modelType: "BAAI/bge-small-en-v1.5",
    quantized: false,
  });

export async function llm() {
    // Load essay from abramov.txt in Node
    const path = "./rc/chiikawa.txt";

    const essay = await fs.readFile(path, "utf-8");
  
    // Create Document object with essay
    const document = new Document({ text: essay, id_: path });
  
    // Split text and create embeddings. Store them in a VectorStoreIndex
    const index = await VectorStoreIndex.fromDocuments([document]);
  
    // Query the index
    const queryEngine = index.asQueryEngine();
  
    console.log(`response: start...`);
    const response = await queryEngine.query({
      query: "Who is the author of Chiikawa?",
    });
  
    // Output response
    console.log(`response: ${JSON.stringify(response.message)}`);
  }

llm().then(() => {console.log(`end`); });