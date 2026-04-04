# AI Novel Generator (Gradio + LM Studio)

A powerful, local-first AI novel generator that leverages LM Studio's OpenAI-compatible API to create immersive stories chapter-by-chapter.

## Features

- **Sequential Generation**: Maintains narrative continuity by including previous chapter context in each new generation.
- **Customizable Structure**: Set your preferred plot seed, number of chapters, and target token length.
- **Multi-language Support**: Select between **Korean**, **Japanese** and **English** for your story.
- **Real-time Streaming**: Watch the AI write your novel in real-time within the Gradio interface.
- **Direct Export**: Automatically bundles all generated chapters into a single `.txt` file for easy download.
- **Sequential Output Storage**: All generated novels are automatically saved in the `output/` folder with incremental numbering (e.g., `novel_001.txt`, `novel_002.txt`), ensuring your work is never overwritten.
- **Configurable System Prompt**: Fine-tune the AI's persona via `system_prompt.txt` or choose from curated presets (Literary, Web Novel, Fantasy, Romance, Sci-Fi).
- **AI-powered Seed Generation**: Instantly brainstorm creative story ideas based on your chosen writing style and language.
- **Flexible Model Selection**: Choose from preset optimized models or enter a custom model identifier.

## Prerequisites

- **Python 3.10+**
- **LM Studio**: Running a local server (default port `1234`).
- **Supported Models**: 
  - `gemma-4-31b-it` (Default)
  - `google/gemma-4-26b-a4b`
  - `qwen/qwen3.5-35b-a3b`
  - `qwen3.5-27b`
  - (Any other custom model identifier)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Python_novel.git
   cd Python_novel
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Windows git bash
   source .venv/Scripts/activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Narrative Generation Workflows

You can generate your novel using two distinct workflows:

### Workflow A: Manual Full-Control (Recommended)
This mode allows you to refine the story's direction before final generation.
1.  **Input Initial Idea**: Enter a brief concept in the "Initial Idea / Seed" box, or click **🎲 Auto-Generate Seed** to let the AI brainstorm a unique starting point for you.
2.  **Generate Plot**: Click **1. Generate Plot Outline**. The AI will create a chapter-by-chapter summary.
3.  **Review & Edit**: **(Crucial Step)** You can manually edit the generated plot in the "2. Editable Plot Outline" box to fix inconsistencies or add specific plot points.
4.  **Start Generation**: Click **3. Start Novel Generation**. The AI will follow your refined plot exactly, chapter by chapter.

### Workflow B: Automated Batch Mode
Perfect for creating multiple variations or generating large volumes of content automatically.
1.  **Input Idea & Batch Count**: Enter your initial idea and the number of independent novels you want to create (e.g., 5).
2.  **Launch**: Click **🚀 Batch Start**.
3.  **Automatic Execution**: The system will automatically:
    - Generate a unique plot outline for each batch iteration.
    - Immediately start generating the novel based on that specific plot.
    - Save each completed novel as a separate `.txt` file in the `output/` directory.

## Configuration & Advanced Settings

### System Prompt (`system_prompt.txt`)
The application automatically loads the initial system prompt from `system_prompt.txt` at startup. 
- Edit this file or the UI text box to define the global persona, tone, and constraints of your AI novelist.
- Use the **System Prompt Presets** dropdown to quickly switch between different writing styles (e.g., Epic Fantasy vs. Web Novel).
- Click the **💾 Save** button next to the system prompt to overwrite the local `system_prompt.txt` with your current settings.

### AI Seed Generation (🎲)
If you're facing writer's block, the **Auto-Generate Seed** feature uses your current system prompt settings to brainstorm a creative concept. 
- It ensures the generated idea matches the specific tone and genre defined in your persona.
- The output is automatically placed into the "Initial Idea / Seed" box, ready for plot generation.

### Generation Parameters
Adjust these in the "⚙️ Generation Parameters" accordion:
- **Temperature**: Higher values (e.g., 1.2) increase creativity, while lower values (e.g., 0.5) make the output more focused and predictable.
- **Top-P**: Controls the diversity of the vocabulary.
- **Repetition Penalty**: Helps prevent the model from repeating the same phrases or sentences.

## Usage

1. **Start LM Studio**:
   - Load your preferred model.
   - Start the "Local Server" on port `1234`.
2. **Launch the Generator**:
   ```bash
   python app.py
   ```
3. **Open the Web UI**:
   - Navigate to [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.
4. **Begin Writing**:
   - Input your plot seed, select the language, and click **Generate Novel**.

## UI Preview

The app features a modern, responsive Gradio interface designed for a seamless writing experience.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
