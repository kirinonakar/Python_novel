# AI Novel Generator (Gradio + LM Studio)

A powerful, local-first AI novel generator that leverages LM Studio's OpenAI-compatible API to create immersive stories chapter-by-chapter.

## Features

- **Sequential Generation**: Maintains narrative continuity by including previous chapter context in each new generation.
- **Customizable Structure**: Set your preferred plot seed, number of chapters, and target token length.
- **Multi-language Support**: Select between **Korean** and **Japanese** for your story.
- **Real-time Streaming**: Watch the AI write your novel in real-time within the Gradio interface.
- **Direct Export**: Automatically bundles all generated chapters into a single `.txt` file for easy download.
- **Sequential Output Storage**: All generated novels are automatically saved in the `output/` folder with incremental numbering (e.g., `novel_001.txt`, `novel_002.txt`), ensuring your work is never overwritten.
- **Configurable System Prompt**: Fine-tune the AI's persona and writing style via `system_prompt.txt`.
- **Flexible Model Selection**: Choose from preset optimized models or enter a custom model identifier.

## Prerequisites

- **Python 3.10+**
- **LM Studio**: Running a local server (default port `1234`).
- **Supported Models**: 
  - `google/gemma-4-26b-a4b` (Default)
  - `gemma-4-31b-it`
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
