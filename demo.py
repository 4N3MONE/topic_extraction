import gradio as gr
from lavis.models import load_model_and_preprocess
import torch
import argparse
import openai
import ast
import json

openai.api_key = '*************'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", default="blip2_vicuna_instruct")
    parser.add_argument("--model-type", default="vicuna7b")
    args = parser.parse_args()

    image_input = gr.Image(type="pil")

    sampling = gr.Radio(
        choices=["Beam search", "Nucleus sampling"],
        value="Beam search",
        label="Text Decoding Method",
        interactive=True,
    )

    top_p = gr.Slider(
        minimum=0.5,
        maximum=1.0,
        value=0.9,
        step=0.1,
        interactive=True,
        label="Top p",
    )

    beam_size = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        interactive=True,
        label="Beam Size",
    )


    option_selector = gr.Dropdown(
        choices=["ad", "animal2human", "describe", "human2animal"],
        label="GPT Prompt Selector",
    )

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('Loading model...')

    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name,
        model_type=args.model_type,
        is_eval=True,
        device=device,
    )

    print('Loading model done!')

def convert_string_to_dict(output_next_prompt):
    try:
        output_dict = ast.literal_eval(output_next_prompt)
    except (ValueError, SyntaxError) as e:
        return output_next_prompt
    return output_dict


def inference(image, option, beam_size, top_p, decoding_method):
    use_nucleus_sampling = decoding_method == "Nucleus sampling"
    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    prompt = "Write an appropriate Instagram post for the given photo."

    samples = {
        "image": image,
        "prompt": prompt,
    }

    output = model.generate(
    samples,
    length_penalty=1,
    repetition_penalty=1,
    num_beams=beam_size,
    max_length=250,
    min_length=1,
    top_p=top_p,
    use_nucleus_sampling=use_nucleus_sampling,
    )

    output = "\n".join([str(elem) for elem in output])
    if option == "ad":
        new_prompt = "다음 지시사항에 맞추어 주어진 영어 텍스트를 한글 인스타그램 포스트 내용으로 바꿔주세요. 결과물은 {'post' : '예시 텍스트', 'hashtag' : '예시 해시태그'}식의 딕셔너리 형태여야 합니다. 이러한 작업 지침은 ChatGPT 모델에 주어지며, ChatGPT 모델이 지침을 완료하는지 평가합니다. " + \
                     "요구 사항은 다음과 같습니다:\n" + \
                     "영어 텍스트 :" + output + "\n" +\
                     "1. ‘post’의 답변은 자연스러우며 어법에 맞아야 합니다.\n" + \
                     "2. ’post’의 내용은 100단어를 넘지 않아야 합니다.\n" + \
                     "3. ‘post’는 영어 텍스트에서 등장하는 동물을 제외한 다른 사물에 관한 광고 게시물이어야 합니다.\n" + \
                     "4. ‘post’의 광고의 대상은 복수의 사물이 아닌 하나의 사물이어야 합니다.\n" + \
                     "5. ‘post’ 에 구매자에게 사물에 관한 장점을 어필하는 부분이 포함되어야 합니다.\n" + \
                     "6. ‘post’에 등장하는 동물이 광고 대상을 마음에 들어할 것이라는 문장이 포함되어야 합니다.\n" + \
                     "7. ‘post’맨 앞에 #광고 라는 문자열이 있어야 합니다.\n" + \
                     "8. ‘post’에는 해시태그가 아닌 본문에 어울리는 내용만 있어야 합니다.\n" + \
                     "9. ‘hashtag’는 본문과 어울리는 해시태그를 2~3개 포함해야 합니다. "
    elif option == "animal2human":
        new_prompt = "다음 지시사항에 맞추어 주어진 영어 텍스트를 한글 인스타그램 포스트 내용으로 바꿔주세요. 결과물은 {'post' : '예시 텍스트', 'hashtag' : '예시 해시태그'}식의 딕셔너리 형태여야 합니다. 이러한 작업 지침은 ChatGPT 모델에 주어지며, ChatGPT 모델이 지침을 완료하는지 평가합니다. " + \
                     "요구 사항은 다음과 같습니다:\n" + \
                      "영어 텍스트 :" + output + "\n" +\
                     "1. ‘post’는 영어 텍스트에서 등장하는 동물이 말을 할 수 있다면 주인에게 하는 말입니다.\n" + \
                     "2. ‘post’는 ‘’로 구분된 하나의 문장을 넘지 말아야 합니다.\n" + \
                     "3. ‘post’는 친근하고 애정을 담아야 하며 존댓말을 하지 말아야 합니다.\n" + \
                     "4. ‘post’는 동물이 사람에게 하는 말 말고는 다른 내용은 제외해야 합니다.\n" + \
                     "5. ‘post’는 자연스러우며 어법에 맞아야 합니다.\n" + \
                     "6. ‘post’에는 해시태그가 아닌 본문에 어울리는 내용만 있어야 합니다.\n" + \
                     "7. ‘hashtag’는 본문과 어울리는 해시태그를 2~3개 포함해야 합니다. "
    elif option == "describe":
        new_prompt = "다음 지시사항에 맞추어 주어진 영어 텍스트를 한글 인스타그램 포스트 내용으로 바꿔주세요. 결과물은 {'post' : '예시 텍스트', 'hashtag' : '예시 해시태그'}식의 딕셔너리 형태여야 합니다. 이러한 작업 지침은 ChatGPT 모델에 주어지며, ChatGPT 모델이 지침을 완료하는지 평가합니다. " + \
                     "요구 사항은 다음과 같습니다:\n" + \
                      "영어 텍스트 :" + output + "\n" +\
                     "1. ‘post’는 사진에 관한 인스타그램 포스트 내용입니다.\n" + \
                     "2. ‘post’의 주된 내용은 사진에 등장하는 동물에 관한 것입니다.\n" + \
                     "3. ‘post’는 단순한 묘사가 아니라 동물에 관한 이야기를 담고 있어야 합니다.\n" + \
                     "4. ‘post’는 자연스러우며 어법에 맞아야 합니다.\n" + \
                     "5. ‘post’에는 해시태그가 아닌 본문에 어울리는 내용만 있어야 합니다.\n" + \
                     "6. ‘hashtag’는 본문과 어울리는 해시태그를 2~3개 포함해야 합니다. "
    else: # Option 4
        new_prompt = "다음 지시사항에 맞추어 주어진 영어 텍스트를 한글 인스타그램 포스트 내용으로 바꿔주세요. 결과물은 {'post' : '예시 텍스트', 'hashtag' : '예시 해시태그'}식의 딕셔너리 형태여야 합니다. 이러한 작업 지침은 ChatGPT 모델에 주어지며, ChatGPT 모델이 지침을 완료하는지 평가합니다. " + \
                     "요구 사항은 다음과 같습니다:\n" + \
                      "영어 텍스트 :" + output + "\n" +\
                     "1. 'post'는 한 문장입니다.\n" + \
                     "2. 'post'는 영어 텍스트 상황에서 애완동물에게 주인이 해주는 말입니다.\n" + \
                     "3. ‘post’는 상황 설명에 대한 내용은 제외합니다.\n" + \
                     "4. ‘post’는 자연스러우며 어법에 맞아야 합니다.\n" + \
                     "5. ‘post’에는 해시태그가 아닌 본문에 어울리는 내용만 있어야 합니다.\n" + \
                     "6. ‘hashtag’는 본문과 어울리는 해시태그를 2~3개 포함해야 합니다. "


    chat_response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "당신은 도움이 필요한 어시스턴트입니다."},
        {"role": "user", "content": new_prompt},
      ]
    )

    output_next_prompt = chat_response['choices'][0]['message']['content']
    output_next_prompt = convert_string_to_dict(output_next_prompt)
    if isinstance(output_next_prompt, dict):
        output_post = output_next_prompt['post']
        output_hashtag = output_next_prompt['hashtag']
    else:
        output_post = output_next_prompt
        output_hashtag = ""
    return f"{output_post}\n\n\n{output_hashtag}"

gr.Interface(
    fn=inference,
    inputs=[image_input, option_selector, beam_size, top_p, sampling],
    outputs="text",
    allow_flagging="never",
).launch(share=True)
