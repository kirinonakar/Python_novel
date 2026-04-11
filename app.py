import gradio as gr
from openai import OpenAI
import os
import re
import time
import threading

# Global state for batch queue
BATCH_QUEUE = []
QUEUE_LOCK = threading.Lock()


def clean_thought_tags(text):
    """
    Remove internal reasoning tags like <|channel>thought ... <channel|>
    commonly used in Gemma 4 and similar reasoning models.
    """
    # Pattern for complete blocks
    text = re.sub(r'<\|channel>thought.*?<channel\|>', '', text, flags=re.DOTALL)
    # Pattern for unclosed blocks at the end of a stream
    text = re.sub(r'<\|channel>thought.*$', '', text, flags=re.DOTALL)
    # Individual tokens that might leak
    text = text.replace('<|channel>thought', '').replace('<channel|>', '')
    return text.strip()
def load_system_prompt(filename="system_prompt.txt"):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return "You are a professional novelist. Write engaging and immersive stories."

def open_output_folder():
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Use os.startfile for Windows, or subprocess for other OS
    try:
        os.startfile(os.path.abspath(output_dir))
    except Exception:
        import subprocess
        import platform
        if platform.system() == "Windows":
            subprocess.run(["explorer", os.path.abspath(output_dir)])
        elif platform.system() == "Darwin":
            subprocess.run(["open", os.path.abspath(output_dir)])
        else:
            subprocess.run(["xdg-open", os.path.abspath(output_dir)])

def save_system_prompt(content, filename="system_prompt.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content.strip())
        return "✅ System prompt saved successfully!"
    except Exception as e:
        return f"❌ Failed to save: {str(e)}"

def summarize_chapter(client, model, chapter_text, language, temperature=0.5):
    prompt = (
        f"Summarize the following chapter in 3-4 sentences in {language}.\n"
        f"Focus only on key plot events and character changes that are essential for continuity.\n\n"
        f"Chapter Content:\n{chapter_text[:4000]}" # Truncate for summary if too long
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

def split_plot_into_chapters(plot_seed, num_chapters):
    import re
    # Look for common chapter markers: Chapter 1, 1장, 제 1장, Chapter 01, etc.
    # We want to find the text between markers.
    chapter_plots = {}
    
    # Try to find all chapter-like markers and their positions
    pattern = re.compile(r'(Chapter\s*\d+|第?\s*\d+\s*[章장])', re.IGNORECASE)
    matches = list(pattern.finditer(plot_seed))
    
    if not matches:
        return None
    
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(plot_seed)
        content = plot_seed[start:end].strip()
        # Extract number from marker
        num_match = re.search(r'\d+', matches[i].group())
        if num_match:
            ch_num = int(num_match.group())
            chapter_plots[ch_num] = content
            
    return chapter_plots

def split_full_text_into_chapters(text, language):
    import re
    # Remove error messages
    text = re.sub(r'\n\n\[Generation Stopped/Error\].*$', '', text, flags=re.DOTALL)
    
    if language == "Korean":
        pattern = r'(?:^|\n)# 제 (\d+)장'
    elif language == "Japanese":
        pattern = r'(?:^|\n)# 第 (\d+) 章'
    else:
        pattern = r'(?:^|\n)# Chapter (\d+)'
        
    matches = list(re.finditer(pattern, text))
    chapters = {}
    for i in range(len(matches)):
        try:
            ch_num = int(matches[i].group(1))
            start = matches[i].end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            chapters[ch_num] = text[start:end].strip()
        except:
            continue
    return chapters

def suggest_next_chapter_fn(text, language):
    chapters = split_full_text_into_chapters(text, language)
    if not chapters:
        return 1
    
    # 챕터라고 인식된 텍스트가 일정 길이(예: 300자) 이상이어야 실제 작성된 챕터로 인정합니다.
    # 이는 줄거리(Outline)나 모델의 할루시네이션으로 인해 챕터 번호만 출력된 경우를 제외하기 위함입니다.
    valid_chapters = [ch for ch, content in chapters.items() if len(content) >= 300]
    
    if not valid_chapters:
        return 1
        
    return max(valid_chapters) + 1

SYSTEM_PROMPT_PRESETS = {
    "Standard / Literary Fiction": 'You are an award-winning, bestselling novelist known for elegant prose, deep psychological insight, and compelling character arcs. \nYour writing style is immersive and vivid. Strictly adhere to the "Show, Don\'t Tell" principle—describe sensory details, actions, and character reactions rather than simply stating emotions. \nMaintain a consistent tone, ensure natural-sounding dialogue, and pace the narrative to keep the reader deeply engaged. Never use meta-commentary or acknowledge that you are an AI.',
    "Web Novel / Light Novel": 'You are a top-ranking web novel author known for highly addictive pacing, dynamic character interactions, and gripping cliffhangers. \nYour writing style is accessible, fast-paced, and highly entertaining. \nUse frequent paragraph breaks to make the text easy to read on mobile devices. Focus heavily on punchy, expressive dialogue and characters\' internal thoughts. Keep the plot moving forward dynamically, and avoid overly dense or tedious descriptions. Every chapter must end in a way that makes the reader desperate to read the next.',
    "Epic / Dark Fantasy": 'You are a master of epic and dark fantasy. You excel at intricate world-building, crafting gritty atmospheres, and writing high-stakes conflicts. \nUse rich, evocative, and sometimes archaic vocabulary to bring the fantasy world to life. Describe the environments, magic systems, and battles with visceral sensory details. Characters should be morally complex and face difficult dilemmas. The tone should be serious, atmospheric, and immersive.',
    "Romance / Emotional Drama": 'You are a bestselling romance and drama author. Your greatest strength lies in capturing the intricate emotional dynamics, chemistry, and romantic tension between characters. \nFocus deeply on micro-expressions, body language, and the unspoken feelings between characters. Write dialogue that is witty, passionate, or emotionally raw, depending on the scene. Build the emotional stakes gradually, making the readers deeply invested in the characters\' relationships.',
    "Sci-Fi / Thriller": 'You are a master of science fiction and suspense thrillers. Your prose is sharp, precise, and gripping. \nFocus on building relentless suspense and a creeping sense of tension. Describe technology, environments, or action sequences with clear, logical, yet cinematic detail. Keep the sentences relatively punchy during action or tense scenes to accelerate the pacing. Leave the readers constantly guessing what will happen next.'
}

def apply_preset(preset_name):
    if preset_name == "Custom (File Default)":
        return load_system_prompt()
    if preset_name in SYSTEM_PROMPT_PRESETS:
        return SYSTEM_PROMPT_PRESETS[preset_name]
    return ""

def generate_random_seed_fn(api_base, model_name, system_prompt, language, temperature=1.0, top_p=0.95):
    # 1. 클릭 즉시 UI에 피드백을 줍니다.
    yield "⏳ 시드를 구상하고 있습니다. 잠시만 기다려주세요..."
    
    if not api_base:
        api_base = "http://localhost:1234/v1"
    if not model_name:
        model_name = "gemma-4-31b-it"
        
    client = OpenAI(base_url=api_base, api_key="lm-studio")
    
    prompt = (
        f"Based on your assigned writing style, genre, and persona in the system prompt, "
        f"brainstorm a highly creative, unique, and engaging initial plot seed (core idea) for a new novel. "
        f"Write the seed in {language}. "
        f"Keep it concise (about 3-5 sentences). "
        f"Output ONLY the plot seed text. "
        f"Do not include titles, greetings, meta-commentary, or any internal reasoning tags like <|channel>thought."
    )
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=float(temperature), 
            top_p=float(top_p),
            max_tokens=2000
        )
        # 2. 결과가 나오면 텍스트 박스를 결과물로 덮어씌웁니다.
        content = clean_thought_tags(response.choices[0].message.content)
        yield content
    except Exception as e:
        error_msg = str(e)
        if "Failed to parse input at pos 0" in error_msg:
            error_hint = "\n💡 [Hint] Model mismatch detected. Ensure LM Studio chat template is correctly set for Gemma 4."
            yield f"❌ [Error] {error_msg}{error_hint}"
        else:
            yield f"❌ [Error] 시드 생성 실패: {error_msg}"

def generate_plot_fn(
    api_base, 
    model_name, 
    system_prompt, 
    plot_seed, 
    num_chapters, 
    language,
    temperature=1.0,
    top_p=0.95,
    repetition_penalty=1.1
):
    if not api_base:
        api_base = "http://localhost:1234/v1"
    if not model_name:
        model_name = "gemma-4-31b-it"
    
    client = OpenAI(base_url=api_base, api_key="lm-studio")
    
    headers = {
        "Korean": [
            "1. 제목",
            "2. 핵심 주제의식과 소설 스타일",
            "3. 등장인물 이름, 설정",
            "4. 세계관 설정",
            "5. 각 장 제목과 내용, 핵심 포인트 (Include clear markers like '제 1장', '제 2장', etc. for each chapter)"
        ],
        "Japanese": [
            "1. タイトル",
            "2. 核心となるテーマと小説のスタイル",
            "3. 登場人物の名前・設定",
            "4. 世界観設定",
            "5. 各章のタイトルと内容、重要ポイント (Include clear markers like '第 1 章', '第 2 章', etc. for each chapter)"
        ],
        "English": [
            "1. Title",
            "2. Core Theme and Novel Style",
            "3. Character Names and Settings",
            "4. World Building/Setting",
            "5. Chapter Titles, Content, and Key Points (Include clear markers like 'Chapter 1', 'Chapter 2', etc. for each chapter)"
        ]
    }
    h = headers.get(language, headers["Korean"])
    
    prompt = (
        f"Based on the following seed, create a detailed plot outline for a {num_chapters}-chapter novel in {language}.\n"
        f"Seed: {plot_seed}\n\n"
        f"FORMAT INSTRUCTIONS:\n"
        f"Please organize the output into the following 5 sections in {language}:\n"
        f"{h[0]}\n"
        f"{h[1]}\n"
        f"{h[2]}\n"
        f"{h[3]}\n"
        f"{h[4]}\n\n"
        f"Ensure every section is detailed and provides a solid foundation for writing the novel.\n"
        f"Output ONLY the plot outline based on this format, without any greetings, meta-commentary, or <|channel>thought blocks."
    )
    
    plot_content = ""
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            timeout=60.0,
            temperature=temperature,
            top_p=top_p
        )
        
        chunk_count = 0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                plot_content += chunk.choices[0].delta.content
                chunk_count += 1
                if chunk_count % 5 == 0:  # Throttling
                    yield clean_thought_tags(plot_content)
        
        yield clean_thought_tags(plot_content)  # Final yield
    except Exception as e:
        error_msg = str(e)
        current_plot = clean_thought_tags(plot_content)
        if "Failed to parse input at pos 0" in error_msg:
            yield current_plot + f"\n\n[Generation Stoppped/Error]: {error_msg}\n\n💡 [Tip] This 'pos 0' error is common with Gemma 4 in LM Studio. Try restarting the model server or checking the Chat Template settings."
        else:
            yield current_plot + f"\n\n[Generation Stoppped/Error]: {error_msg}"

def refine_plot_fn(
    api_base, 
    model_name, 
    system_prompt, 
    current_plot, 
    num_chapters, 
    language,
    temperature=1.0,
    top_p=0.9,
    repetition_penalty=1.1
):
    if not api_base:
        api_base = "http://localhost:1234/v1"
    if not model_name:
        model_name = "gemma-4-31b-it"
    
    client = OpenAI(base_url=api_base, api_key="lm-studio")
    
    headers = {
        "Korean": [
            "1. 제목",
            "2. 핵심 주제의식과 소설 스타일",
            "3. 등장인물 이름, 설정",
            "4. 세계관 설정",
            "5. 각 장 제목과 내용, 핵심 포인트 (Ensure clear chapter markers like '제 1장', '제 2장', etc. are preserved)"
        ],
        "Japanese": [
            "1. タイトル",
            "2. 核心となるテーマと小説のスタイル",
            "3. 登場人物の名前・設定",
            "4. 世界観設定",
            "5. 各章のタイトルと内容、重要ポイント (Ensure clear chapter markers like '第 1 章', '第 2 章', etc. are preserved)"
        ],
        "English": [
            "1. Title",
            "2. Core Theme and Novel Style",
            "3. Character Names and Settings",
            "4. World Building/Setting",
            "5. Chapter Titles, Content, and Key Points (Ensure clear chapter markers like 'Chapter 1', 'Chapter 2', etc. are preserved)"
        ]
    }
    h = headers.get(language, headers["Korean"])

    prompt = (
        f"You are a master story architect. Your task is to refine and elaborate on the following plot outline for a {num_chapters}-chapter novel in {language}.\n\n"
        f"[Current Plot Outline]\n{current_plot}\n\n"
        f"FORMAT & REFINEMENT INSTRUCTIONS:\n"
        f"Please refine the plot while STRICTLY maintaining the following 5-section format in {language}:\n"
        f"{h[0]}\n"
        f"{h[1]}\n"
        f"{h[2]}\n"
        f"{h[3]}\n"
        f"{h[4]}\n\n"
        f"REFINEMENT GOALS:\n"
        f"- Polish content for better emotional resonance and logical consistency.\n"
        f"- Add vivid sensory details and deeper character motivations.\n"
        f"- Ensure the {int(num_chapters)}-chapter pacing is dynamic and leading toward a powerful climax.\n"
        f"Output ONLY the refined plot text, without any greetings, meta-talk, or internal reasoning tags like <|channel>thought."
    )
    
    refined_content = ""
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            timeout=60.0,
            temperature=temperature,
            top_p=top_p
        )
        
        chunk_count = 0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                refined_content += chunk.choices[0].delta.content
                chunk_count += 1
                if chunk_count % 5 == 0:
                    yield clean_thought_tags(refined_content)
        
        yield clean_thought_tags(refined_content)
    except Exception as e:
        error_msg = str(e)
        current_refined = clean_thought_tags(refined_content)
        yield current_refined + f"\n\n[Refinement Stopped/Error]: {error_msg}"

def get_next_filename(directory, prefix="novel_", extension=".txt"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    max_num = 0
    for f in files:
        try:
            num_str = f[len(prefix):-len(extension)]
            num = int(num_str)
            if num > max_num:
                max_num = num
        except ValueError:
            continue
    
    next_num = max_num + 1
    return os.path.join(directory, f"{prefix}{next_num:03d}{extension}")

def generate_novel(
    api_base, 
    model_name, 
    system_prompt, 
    plot_seed, 
    num_chapters, 
    target_tokens, 
    language,
    start_chapter=1,
    existing_content="",
    temperature=1.0,
    top_p=0.95,
    repetition_penalty=1.1
):
    if not api_base:
        api_base = "http://localhost:1234/v1"
    if not model_name:
        model_name = "gemma-4-31b-it"
    
    client = OpenAI(base_url=api_base, api_key="lm-studio")
    
    full_text = ""
    chapter_summaries = []
    
    # Resuming logic
    start_chapter = int(start_chapter)
    if start_chapter > 1 and existing_content:
        yield "🔄 Resuming generation... Reconstructing context from existing chapters. Please wait.", None
        chapters_map = split_full_text_into_chapters(existing_content, language)
        
        rebuilt_text = ""
        for ch in range(1, start_chapter):
            if language == "Korean": header = f"\n\n# 제 {ch}장\n\n"
            elif language == "Japanese": header = f"\n\n# 第 {ch} 章\n\n"
            else: header = f"\n\n# Chapter {ch}\n\n"
            
            content = chapters_map.get(ch, "")
            if content:
                rebuilt_text += header + content
                summary = summarize_chapter(client, model_name, content, language)
                chapter_summaries.append(summary)
            else:
                # If a chapter is missing, we still need to keep the order for summaries
                chapter_summaries.append("")
        full_text = rebuilt_text
        yield full_text + "\n\n✅ Context reconstructed. Starting generation...", None
    
    # Pre-parse chapters from plot if possible
    chapter_plots = split_plot_into_chapters(plot_seed, int(num_chapters))
    
    for ch in range(start_chapter, int(num_chapters) + 1):
        # Build improved prompt for the current chapter
        prompt = f"You are a professional novelist writing a novel in {language}.\n\n"
        prompt += f"[Book Information]\n"
        prompt += f"- Total Chapters: {int(num_chapters)}\n"
        prompt += f"- Current Chapter to Write: Chapter {ch}\n"
        
        # Specific plot for this chapter
        if chapter_plots and ch in chapter_plots:
            prompt += f"- Current Chapter Plot: {chapter_plots[ch]}\n\n"
        else:
            prompt += f"- Master Plot Outline:\n{plot_seed}\n\n"
        
        # Context Management
        if chapter_summaries:
            prompt += "[Story So Far (Summary)]\n"
            for i, summ in enumerate(chapter_summaries):
                prompt += f"Chapter {i+1}: {summ}\n"
            prompt += "\n"
            
            # Direct previous context (last 1000 chars)
            last_content = full_text.split("\n\n#")[-1] # Rudimentary way to get last chapter text
            prompt += f"[Directly Preceding Content (End of Chapter {ch-1})]\n"
            prompt += f"\"{last_content[-1000:]}\"\n\n"
            
        prompt += f"CRITICAL INSTRUCTION:\n"
        prompt += f"1. Write ONLY Chapter {ch}. Do not rush into future chapters.\n"
        prompt += f"2. Target length: ~{int(target_tokens)} tokens.\n"
        prompt += f"3. Output ONLY the story text. No meta-talk or phrases like 'Sure, here is chapter...'.\n"
        prompt += f"4. NEVER use internal reasoning tags, thinking blocks, or <|channel>thought tokens."

        try:
            chapter_text = ""
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max(int(target_tokens) + 1000, 4096),
                stream=True,
                timeout=180.0,
                temperature=temperature,
                top_p=top_p
            )
            
            if language == "Korean":
                current_chapter_title = f"\n\n# 제 {ch}장\n\n"
            elif language == "Japanese":
                current_chapter_title = f"\n\n# 第 {ch} 章\n\n"
            else:
                current_chapter_title = f"\n\n# Chapter {ch}\n\n"
            full_text += current_chapter_title
            yield full_text, None
            
            chunk_count = 0
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chapter_text += content
                    chunk_count += 1
                    # Yield more frequently at the start, then throttle to reduce lag
                    if chunk_count < 10 or chunk_count % 10 == 0:
                        yield full_text + clean_thought_tags(chapter_text), None
            
            full_text += clean_thought_tags(chapter_text)
            yield full_text, None # Ensure end of chapter is shown
            
            # Summarize this chapter for future context
            summary = summarize_chapter(client, model_name, chapter_text, language)
            chapter_summaries.append(summary)
            
        except Exception as e:
            error_msg = str(e)
            current_text = full_text + clean_thought_tags(chapter_text)
            hint = ""
            if "Failed to parse input at pos 0" in error_msg:
                hint = "\n\n💡 [Tip] Gemma 4 parsing error (pos 0). This usually requires adjusting the Chat Template in LM Studio to support reasoning tokens or disabling prompt caching."
            
            yield current_text + f"\n\n[Generation Stopped/Error]: {error_msg}{hint}", None
            break

    # Save to file in output directory with sequential numbering
    output_dir = "output"
    file_path = get_next_filename(output_dir)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    yield full_text, file_path

def batch_process(
    api_base, 
    model_name, 
    system_prompt, 
    plot_seed, 
    num_chapters, 
    target_tokens, 
    language, 
    batch_count,
    temperature=1.0,
    top_p=0.95,
    repetition_penalty=1.1
):
    batch_count = int(batch_count)
    for i in range(batch_count):
        # 1. Generate Plot
        plot_content = ""
        plot_gen = generate_plot_fn(
            api_base, model_name, system_prompt, plot_seed, num_chapters, language,
            temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty
        )
        for p in plot_gen:
            plot_content = p
            yield plot_content, f"Batch {i+1}/{batch_count} - Generating plot...", None
            
        # 2. Generate Novel from that plot (with auto-resume)
        current_novel_text = ""
        current_novel_file = None
        target_ch = int(num_chapters)
        
        while True:
            next_ch = suggest_next_chapter_fn(current_novel_text, language)
            if next_ch > target_ch:
                break
                
            novel_gen = generate_novel(
                api_base, model_name, system_prompt, plot_content, num_chapters, target_tokens, language,
                start_chapter=next_ch,
                existing_content=current_novel_text,
                temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty
            )
            
            for n_text, n_file in novel_gen:
                current_novel_text = n_text
                current_novel_file = n_file
                status_prefix = f"### [Batch {i+1}/{batch_count} In Progress]\n"
                if next_ch > 1:
                    status_prefix += f"**(Auto-Resuming from Chapter {next_ch})**\n\n"
                else:
                    status_prefix += "\n"
                yield plot_content, status_prefix + n_text, n_file
            
            # Check if progress was made to avoid infinite loop on persistent errors
            new_next_ch = suggest_next_chapter_fn(current_novel_text, language)
            if new_next_ch <= next_ch:
                # If no new chapter was successfully added, we stop to prevent infinite retries
                # unless we want to try a few times. For now, let's just break.
                break
            
            # Small delay before resuming
            if new_next_ch <= target_ch:
                time.sleep(2)

def enqueue_batch(*args):
    with QUEUE_LOCK:
        BATCH_QUEUE.append(args)
        return len(BATCH_QUEUE)

def run_batch_worker():
    while True:
        with QUEUE_LOCK:
            if not BATCH_QUEUE:
                break
            params = BATCH_QUEUE.pop(0)
        
        # Run the existing batch_process generator with captured parameters
        # batch_process returns (plot_output, output_text, download_link)
        for p, t, f in batch_process(*params):
            with QUEUE_LOCK:
                current_count = len(BATCH_QUEUE)
            yield p, t, f, current_count
    
    # Final yield to ensure queue count is 0 at the very end
    yield gr.skip(), gr.skip(), gr.skip(), 0

# Gradio Interface
with gr.Blocks(title="AI Novel Generator") as demo:
    gr.Markdown("# 🖋️ AI Novel Generator (LM Studio)")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_base = gr.Textbox(label="LM Studio API Base", value="http://localhost:1234/v1")
            model_name = gr.Dropdown(
                label="Model Name", 
                choices=[
                    "gemma-4-31b-it",                     
                    "google/gemma-4-26b-a4b", 
                    "qwen/qwen3.5-35b-a3b", 
                    "qwen3.5-27b"
                ],
                value="gemma-4-31b-it",
                allow_custom_value=True
            )
            with gr.Row():
                system_prompt_preset = gr.Dropdown(
                    choices=["Custom (File Default)"] + list(SYSTEM_PROMPT_PRESETS.keys()), 
                    label="System Prompt Presets", 
                    value="Custom (File Default)",
                    scale=4
                )
                save_prompt_btn = gr.Button("💾 Save", scale=1)
            
            system_prompt = gr.Textbox(
                label="System Prompt Content", 
                value=load_system_prompt(),
                lines=5
            )
            
            system_status = gr.Textbox(label="Save Status", interactive=False, placeholder="Status will appear here...")
            language = gr.Radio(["Korean", "Japanese", "English"], label="Language", value="Korean")
            
        with gr.Column(scale=1):
            with gr.Row():
                plot_seed = gr.Textbox(label="Initial Idea / Seed", placeholder="Enter the main idea or auto-generate...", lines=3, scale=4)
                auto_seed_btn = gr.Button("🎲 Auto-Generate Seed", scale=1)
            
            num_chapters = gr.Number(label="Number of Chapters", value=5, precision=0)
            target_tokens = gr.Number(label="Target Tokens per Chapter", value=2000, precision=0)
            with gr.Row():
                start_chapter = gr.Number(label="Start Chapter (for Resume)", value=1, precision=0, scale=4)
                find_ch_btn = gr.Button("🔍 Find Next", scale=1)
            
            with gr.Accordion("⚙️ Generation Parameters", open=False):
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5, step=0.1, value=1.0)
                top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.05, value=0.95)
                repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.1)

            gr.Markdown("---")
            gr.Markdown("### 📦 Batch Mode")
            with gr.Group():
                with gr.Row():
                    batch_count = gr.Number(label="Batch Count", value=1, precision=0, minimum=1)
                    queue_count_display = gr.Number(label="Queue", value=0, precision=0, interactive=False, scale=1)
                with gr.Row():
                    batch_start_btn = gr.Button("🚀 Batch Start", variant="primary")
                    batch_stop_btn = gr.Button("⏹️ Stop", variant="stop")
            
    with gr.Row():
        plot_btn = gr.Button("1. Generate Plot Outline", variant="secondary")
        refine_plot_btn = gr.Button("✨ Refine Plot", variant="secondary")
        stop_plot_btn = gr.Button("Stop Plot", variant="stop")
        
    plot_output = gr.Textbox(label="2. Editable Plot Outline (Review and Modify)", lines=10, interactive=True)
    
    with gr.Row():
        generate_btn = gr.Button("3. Start Novel Generation", variant="primary")
        stop_btn = gr.Button("Stop Novel", variant="stop")
    
    output_text = gr.Textbox(label="4. Generated Novel Content", lines=20, interactive=False)
    with gr.Row():
        download_link = gr.File(label="5. Download Full Novel (.txt)", scale=4)
        open_folder_btn = gr.Button("📂 Open Output Folder", scale=1)

    # Preset change event
    system_prompt_preset.change(
        fn=apply_preset,
        inputs=[system_prompt_preset],
        outputs=[system_prompt]
    )

    # Save prompt event
    save_prompt_btn.click(
        fn=save_system_prompt,
        inputs=[system_prompt],
        outputs=[system_status]
    )

    # Plot click event
    plot_click = plot_btn.click(
        fn=generate_plot_fn,
        inputs=[
            api_base, model_name, system_prompt, plot_seed, num_chapters, language,
            temperature, top_p, repetition_penalty
        ],
        outputs=[plot_output]
    )
    
    refine_click = refine_plot_btn.click(
        fn=refine_plot_fn,
        inputs=[
            api_base, model_name, system_prompt, plot_output, num_chapters, language,
            temperature, top_p, repetition_penalty
        ],
        outputs=[plot_output]
    )
    
    stop_plot_btn.click(fn=None, inputs=None, outputs=None, cancels=[plot_click, refine_click])
    
    # Auto-generate seed event
    auto_seed_btn.click(
        fn=generate_random_seed_fn,
        inputs=[
            api_base, model_name, system_prompt, language, 
            temperature, top_p
        ],
        outputs=[plot_seed]
    )
    
    # Novel click event
    novel_click = generate_btn.click(
        fn=generate_novel,
        inputs=[
            api_base, model_name, system_prompt, plot_output, num_chapters, target_tokens, language,
            start_chapter, output_text,
            temperature, top_p, repetition_penalty
        ],
        outputs=[output_text, download_link]
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[novel_click])

    def clear_batch_queue():
        with QUEUE_LOCK:
            BATCH_QUEUE.clear()
        return 0

    # Batch click event with queueing
    batch_event = batch_start_btn.click(
        fn=enqueue_batch,
        inputs=[
            api_base, model_name, system_prompt, plot_seed, num_chapters, target_tokens, language, batch_count,
            temperature, top_p, repetition_penalty
        ],
        outputs=[queue_count_display],
        concurrency_limit=None
    ).then(
        fn=run_batch_worker,
        inputs=None,
        outputs=[plot_output, output_text, download_link, queue_count_display],
        concurrency_limit=1
    )
    
    batch_stop_btn.click(
        fn=clear_batch_queue, 
        inputs=None, 
        outputs=[queue_count_display], 
        cancels=[batch_event]
    )

    # Find next chapter event
    find_ch_btn.click(
        fn=suggest_next_chapter_fn,
        inputs=[output_text, language],
        outputs=[start_chapter]
    )

    # Open folder event
    open_folder_btn.click(
        fn=open_output_folder,
        inputs=[],
        outputs=[]
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, theme=gr.themes.Soft())
