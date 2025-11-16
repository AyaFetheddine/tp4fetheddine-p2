package ma.emsi.fetheddine.tp4fetheddine_p2.llm;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;

/**
 * Service d'accès centralisé au modèle de langage Gemini via LangChain4j.
 * Gère la mémoire de chat et le rôle système pour maintenir le contexte.
 */
@Dependent
public class LlmClient implements Serializable {


    private transient ChatMemory chatMemory;
    private transient Assistant assistant;
    private String systemRole;




    public LlmClient() {

        // Lecture de la clé API depuis les variables d'environnement
        String envKey = System.getenv("GEMINI_KEY");
        if (envKey == null || envKey.isBlank()) {
            throw new IllegalStateException(
                    "Erreur : variable d'environnement GEMINI_KEY absente ou vide."
            );
        }


        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(envKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .build();


        chatMemory = MessageWindowChatMemory.withMaxMessages(10);


        assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .build();
    }

    public void setSystemRole(String role) {
        this.systemRole = role;
        chatMemory.clear();

        if (role != null && !role.trim().isEmpty()) {
            chatMemory.add(SystemMessage.from(role));
        }
    }


    public String ask(String prompt) {
        return assistant.chat(prompt);
    }

    // === Getters ===

    public String getSystemRole() {
        return systemRole;
    }
}