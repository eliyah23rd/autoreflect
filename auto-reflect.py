'''
The purpose of this module is to explore methods for GPT to determine the
path of its own prompt chaining

The code uses a batch of around 100 short-ish philosophy-related posts.

The goal is to replace the large amount of hard-wired code that
determined the stages of the propmt chaining with generic code
that learns the sequence of calls to make.

Ideally it should be GPT itself that decides which stage to execute
next.

'''
import os
from time import sleep
import pickle
import numpy as np
import threading
import asyncio
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

'''
Configuration params
'''
cb_test_async = False
cb_embed_posts = False
cb_chat = True

c_gpt_engine = 'gpt-3.5-turbo'
c_gpt_embed_engine = 'text-embedding-ada-002'

c_num_retries = 5
c_openai_timeout = 180
c_embed_len = 1536
c_batch_size = 50
c_posts_batch_size = 40
c_num_closest = 3 # 10

'''
This is the core function that gets a response from the ChatGPT API

It is written as an async function, so that it can be called multiple times simultaneouslly

'''
async def chatgpt_req(  system_role, prompt='', engine=c_gpt_engine, temp=0.7, 
                        top_p=1.0, tokens=256, freq_pen=0.0, pres_pen=0.0, stop=["\n"]):
    system_role = system_role.encode(encoding='ASCII',errors='ignore').decode()
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    lmessages = [{"role": "system", "content": system_role}]
    if len(prompt) > 0:
        lmessages.append( {"role": "user", "content": prompt})
    response = await openai.ChatCompletion.acreate(
        model=c_gpt_engine,
        messages=lmessages,
        temperature=temp,
        max_tokens=tokens,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        # stop=stop
    )
    text = response['choices'][0]['message']['content'].strip()
    return text

'''
A simple function that enables using timeout when getting the vector embedding from
OpenAI. For some reason, this requests sometimes times out.
'''
def raise_timeout():
  global b_timed_out
  b_timed_out = True
  print('raising timeout')
  raise TimeoutError


def get_embeds(ltexts):
    for itry in range(c_num_retries):
        global b_timed_out
        b_timed_out = False
        timer = threading.Timer(c_openai_timeout, raise_timeout)
        timer.start()
        try:
            response = openai.Embedding.create(
                model=c_gpt_embed_engine,
                input=ltexts
            )
            timer.cancel()
            if b_timed_out:
                print(f'uncaught timeout error on try {itry}')
                continue
            return [response["data"][i]['embedding'] for i in range(len(ltexts))]
        except openai.error.RateLimitError:
            timer.cancel()
            sleep(5)
            return get_embeds(ltexts)
        except TimeoutError as e:
            print(f'timeout error on try {itry}')
            continue
        except openai.APIError:
            timer.cancel()
            print(f'api error on try {itry}')
            continue
        except openai.InvalidRequestError:
            timer.cancel()
            print(f'api invalid request error on try {itry}')
            continue
        # except:
        #     print(f'generic unrecognised error on try {itry}')
        #     continue

    # return [np.zeros((len(ltexts), c_embed_len))]
    return [[0.0] * c_embed_len]

'''
Retrieves the posts from the pickle file
'''

def get_posts_texts(posts_fname):
    with open(f'{posts_fname}.pkl', 'rb') as fh:
        lposts = pickle.load(fh)
    pass
    lpost_texts = []
    for ipost, apost  in enumerate(lposts):
        lpost_texts.append(apost['text'])
    return lposts, lpost_texts

'''
Creates vector embeddings for each of the posts
'''
def make_post_embeds(post_fname):
    _, lpost_texts = get_posts_texts(post_fname)
    num_posts = len(lpost_texts)
    nd_embeds = np.zeros((num_posts, c_embed_len),\
            dtype=np.float32)
    for ipost_start in range(0, num_posts, c_posts_batch_size):
        lbatch = []
        for ipost in range(ipost_start, ipost_start + c_posts_batch_size):
            if ipost >= num_posts:
                break
            lbatch.append(lpost_texts[ipost])
        nd_embeds[ipost_start:ipost_start+len(lbatch)][:] = \
                get_embeds(lbatch)
    np.save(f'{post_fname}_embeds.npy', nd_embeds)

'''
Basic ansync function that allows the app to make multiple simultaneous requests
using the OpenAI API but waits till all have come in before returning
'''
async def gather_answers(lroles, lprompts):
    if type(lroles) is list:
        tasks = [chatgpt_req(role, prompt) for role, prompt in zip(lroles, lprompts)]
    else:
        tasks = [chatgpt_req(lroles, prompt) for prompt in lprompts]
    pass
    # gather waits till all the tasks have completed
    lanswers = await asyncio.gather(*tasks)
    return lanswers

'''
Simple test function to check the functioning of gather
'''
async def test_async():
    prompts = ["What is the meaning of life?", "How do I make a good cup of coffee?", "What is the capital of France?"]
    system_role = "Your job is to provide concise answers to the user\'s questions."
    tasks = [chatgpt_req(system_role, prompt) for prompt in prompts]
    answers = await asyncio.gather(*tasks)
    print(answers)

'''
This is the complete cosine similarity function, even though in this case the vectors are already normalized
'''
def cosine_similarity(vec1, vec2):
    # calculate dot product
    dot_product = vec2 @ vec1
    # calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2, axis=1)
    # calculate cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity

'''
verify_relevant is the second level check after cosine similarity that the text of
the post actually answers the question.
'''
async def verify_relevant(idxs_best, lpost_texts, question):
    prompts = [f'Post:\n{lpost_texts[idx]}\nQuestion:\n{question}' for idx in idxs_best]
    system_role = "Your job is to decide whether the question that appears in the \"Question\" \
            section of the user role is answered by any of the text that appears in the \"Post\" section \
            of the user. \n \
            The text in the Post section is one of the posts written by Eliyah23rd. \
            Please provide only a number from 1 to 10 where 1 indicates that the question is not answered \
            at all in the post section and 10 indicates that the question is addressed directly.\n \
            It is very important that you only provide a number from 1 to 10 and no other words in your answer."
    tasks = [chatgpt_req(system_role, prompt) for prompt in prompts]
    lanswers = await asyncio.gather(*tasks)
    return lanswers

'''
list_posts allows the user to read the posts and returns to the main loop.
If an invalid number is input by the user, we will still return to the main loop.
'''
def list_posts(lposts):
    print('Here are the titles of the posts.')
    for ipost, post in enumerate(lposts):
        title = lposts[ipost]['title']
        print(f'{ipost+1}. {title}')
    try:
        i_sel_post = int(input('Please enter the number of the post you\'d like to see.\n'))
    except ValueError:
        print('I\'m sorry, that is not a valid number.')
        return
    if i_sel_post < 1 or i_sel_post > len(lposts):
        print('I\'m sorry, that is not a valid number.')
        return
    print(lposts[i_sel_post-1]['text'])
    return

'''
Core chat loop
'''
async def chat(post_fname):
    nd_embeds = np.load(f'{post_fname}_embeds.npy')
    l_b_valid_embeds = [val > 0.9 for val in np.linalg.norm(nd_embeds, axis=1)]
    lposts, lpost_texts = get_posts_texts(post_fname)
    num_posts = len(lpost_texts)
    while True:
        question = input('Please enter a question about Eliyah\'s posts or type LIST to see posts or END to exit.\n')
        if question.lower() == "end":
            break
        if question.lower() == "list":
            list_posts(lposts[:-3])
        nd_qembed = get_embeds([question])[0]
        nd_scores = cosine_similarity(nd_qembed, nd_embeds)
        nd_idxs_best = nd_scores.argsort()[-c_num_closest:]
        l_score_strs = await verify_relevant(nd_idxs_best, lpost_texts, question)
        nd_relevance_scores = np.zeros(len(l_score_strs))
        for iscore, score_str in enumerate(l_score_strs):
            try:
                nd_relevance_scores[iscore] = int(score_str)
            except ValueError:
                nd_relevance_scores[iscore] = -1
        i_argmax = np.argmax(nd_relevance_scores)
        if nd_relevance_scores[i_argmax] < 5:
            response = await chatgpt_req('Your job is to inform the user that none of Eliyah\'s posts answer the question.')
            print(response)
            continue
        idx_best_post = nd_idxs_best[i_argmax]
        quote = await chatgpt_req('Your job is to extract all sections from the text the appears \
                in the \"Post\" section of the user content that would provide \
                an answer to the user\'s question in the \"Question\" section.',
                f'Post:\n{lpost_texts[idx_best_post]}\nQuestion:\n{question}')
        response = await chatgpt_req('You are a helpful assistant. Please read all the text the appears \
                in the \"Background\" section of the user content and answer the \
                user\'s question in the \"Question\" section. It is very important that you do not use the words \"Background\" or \"Question\" in your response',
                f'Background:\n{quote}\nQuestion:\n{question}')
        score_str = await chatgpt_req('Your job is to determine whether the answer provided in \
                \"Answer\" section is supported by the text in the \"Background\" \
                section. For reference the original question appears in the \
                \"Question\" section. \
                Please provide only a number from 1 to 10 where 1 indicates that the question is not answered \
                at all in the post section and 10 indicates that the question is addressed directly.\n \
                It is very important that you only provide a number from 1 to 10 and no other words in your answer. '
                f'Background:\n{quote}\nQuestion:\n{question}')
        try:
            score = int(score_str)
        except ValueError:
            score = -1
        if score < 4:
            response = await chatgpt_req('Your job is to inform the user that you cannot answer the question.')
        print(response)
            

'''
The purpose of the main is to allow different uses of the command line.
Rather than parse user options, it is assumed that the programmer is in charge
of the running and can easily go into the code and specify which option they
want to run.
'''
def main():
    if cb_test_async:
        asyncio.run(test_async())
    elif cb_embed_posts:
        make_post_embeds('philposts')
    elif cb_chat:
        asyncio.run(chat('philposts'))
    else:
        print('One of the flow options must be set.')
        exit(1)
    


if __name__ == '__main__':
    main()
    print('done')


