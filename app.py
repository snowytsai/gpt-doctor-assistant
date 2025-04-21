import streamlit as st
import openai
import tiktoken

# 請替換成你自己的 OpenAI API Key
openai.api_key = "你的API金鑰"

# Streamlit 標題
st.title("醫師 GPT 問診助理")

# 使用者輸入（病患主訴 + 醫病對話）
system_prompt = "你是一位內科醫師助理，請根據病患主訴與對話，提供給醫師建議：可能診斷、應追問的問題、建議檢查與治療方案。"

user_input = st.text_area("請輸入病患主訴與對話：", height=200)

if st.button("生成建議"):
    if user_input:
        with st.spinner("GPT 生成中..."):
            # 計算 Token
            encoding = tiktoken.encoding_for_model("gpt-4")
            input_tokens = len(encoding.encode(system_prompt + user_input))

            # 呼叫 GPT
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0.5
            )

            reply = response['choices'][0]['message']['content']
            output_tokens = response['usage']['completion_tokens']
            total_tokens = response['usage']['total_tokens']

            st.subheader("GPT 建議：")
            st.write(reply)

            # 成本計算
            input_cost = input_tokens / 1000 * 0.01
            output_cost = output_tokens / 1000 * 0.03
            total_cost = input_cost + output_cost

            st.info(f"輸入 Token：{input_tokens}，輸出 Token：{output_tokens}")
            st.success(f"估算費用：約 ${total_cost:.4f} 美元")

from streamlit_audiorecorder import audiorecorder
import tempfile
import base64

st.subheader("🎙️ 錄音輸入")

audio = audiorecorder("點我開始錄音", "錄音中...")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")

    # 將錄音寫入暫存檔
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        audio.export(tmp_file.name, format="wav")
        audio_path = tmp_file.name

    st.info("✅ 錄音完成，開始轉文字...")

    import openai
    openai.api_key = "你的API金鑰"

    # 使用 Whisper 將語音轉成文字
    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)

    st.success("語音轉文字結果：")
    st.write(transcript["text"])
    
    # 自動填入主訴欄位（可選）
    user_input = transcript["text"]
