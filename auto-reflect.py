'''
The purpose of this module is to explore methods for GPT to determine the
path of its own prompt chaining

The code uses a batch of around 100 short-ish philosophy-related posts.

The goal is to replace the large amount of hard-wired code that
determined the stages of the propmt chaining with generic code
that learns the sequence of calls to make.

Ideally it should be GPT itself that decides which stage to execute
next.

At this stage of the project all we are doing is making asking GPT
to make just one decision about whether to skip the verification step.

The code includes the first atttempt to define the GPT modules using
configuration files. A complete conversion to this modular and configurable
style will require a redesign of all the code.
For now the code does not look very pretty becuase it is not modular.

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
c_num_closest = 3 # should be 10

fname_extra_history = 'extra_history.txt'
'''
This is the core function that gets a response from the ChatGPT API

It is written as an async function, so that it can be called multiple times simultaneously

'''
async def chatgpt_req(  system_role, lprompts=[], engine=c_gpt_engine, temp=0.7, 
                        top_p=1.0, tokens=256, freq_pen=0.0, pres_pen=0.0, stop=["\n"]):
    system_role = system_role.encode(encoding='ASCII',errors='ignore').decode()
    lprompts = [(role, content.encode(encoding='ASCII',errors='ignore').decode()) for role, content in lprompts]
    lmessages = [{"role": "system", "content": system_role}]
    if len(lprompts) > 0:
        lmessages += [{"role": role, "content": prompt} for role, prompt in lprompts]
    response = await openai.ChatCompletion.acreate(
        model=c_gpt_engine,
        messages=lmessages,
        temperature=temp,
        max_tokens=tokens,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        # stop=stop # ChatCompletion seems to have disabled this option for now and produces an error if you include it.
    )
    text = response['choices'][0]['message']['content'].strip().strip('.')
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

'''
Function that embeds any text as a vector of some 1500 floats.
Used to build the initial database and to embed any user query.
'''
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
        # Keep the following disable because it masks real problems and
        # keyboard ^C but use if you have a very large batch that you
        # don't want to fail under any circumstances.
        # except:
        #     print(f'generic unrecognised error on try {itry}')
        #     continue

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
Basic async function that allows the app to make multiple simultaneous requests
using the OpenAI API but waits till all have come in before returning
'''
async def gather_answers(system_def, lmsgs):
    tasks = [chatgpt_req(system_def, msgs) for msgs in lmsgs]
     # gather waits till all the tasks have completed
    lanswers = await asyncio.gather(*tasks)
    return lanswers

'''
Simple test function to check the functioning of gather
'''
async def test_async():
    prompts = [ ("user", "What is the meaning of life?"), 
                ("user", "How do I make a good cup of coffee?"), 
                ("user", "What is the capital of France?")]
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
This functions parses a step file which consists of section titles which start with '>>'
and the content of that section which cosists of all the rows following the title
up till the next title.
Function produces a dictionary with the titles as keys and content as value.
'''
def parse_step_file(file_name) -> dict:
    sections = {}
    current_section = None
    with open('steps/'+file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>>'):
                current_section = line[2:]
                sections[current_section] = ' '
            elif current_section is not None:
                sections[current_section] += line + ' '
    return sections

'''
Sample verfication file used to make sure that the GPT module returned exactly the
format required.
In this case requires a response of yes or no only but can tolerate a '.' at the end
'''
def yes_or_no(from_gpt):
    if from_gpt.lower() in ['yes', 'no']:
        return True, from_gpt.lower()
    return False, from_gpt

'''
Helper function that builds the gpt response and makes one extra try
in case the GPT response was not in exactly the requested format.
'''
async def get_gpt_response(file_name, input, iter=0):
    step_file_sections = parse_step_file(file_name)
    response = await chatgpt_req(step_file_sections['system'], input)
    bvalid, response = eval(step_file_sections['verify'].strip() + '(\'' + response + '\')')
    if iter > 0 or bvalid:
        return response
    else:
        input += [('assistant', response), ('user', step_file_sections['on_fail'])]
        return await get_gpt_response(file_name, input, iter=iter+1)


'''
verify_relevant is the second level check after cosine similarity that the text of
the post actually answers the question.
'''
async def verify_relevant(nd_scores, idxs_best, lpost_texts, question):
    '''
    For now this is the only place that we give GPT the choice as to whether to add a 
    self-reflection step.
    This is driven by a configuration file called a step file.
    This specifies the role of the module and the required output format
    We put the history of previous decisions as well as the feedback into a History section
    '''
    with open(fname_extra_history, 'r') as f: history = f.read()
    should_we = await get_gpt_response('verify_input_step.txt', 
            [('user', f'Question:\n{question}\nHistory:\n\{history}')])
    '''
    If GPT does not want to execute the next step, we skip the rest of the function.
    In this specific case we revert to the similarity scores for deciding which 
    text to prepend when generating the response to the user
    '''
    if should_we.lower() != 'yes':
        return (nd_scores[idxs_best] * 10).tolist(), False
    lmsgs = [[("user", f'Post:\n{lpost_texts[idx]}\nQuestion:\n{question}')] for idx in idxs_best]
    system_role = "Your job is to decide whether the question that appears in the \"Question\" \
            section of the user role is answered by any of the text that appears in the \"Post\" section \
            of the user. \n \
            The text in the Post section is one of the posts written by Eliyah23rd. \
            Please provide only a number from 1 to 10 where 1 indicates that the question is not answered \
            at all in the post section and 10 indicates that the question is addressed directly.\n \
            It is very important that you only provide a number from 1 to 10 and no other words in your answer."
    tasks = [chatgpt_req(system_role, msgs) for msgs in lmsgs]
    lanswers = await asyncio.gather(*tasks)
    return lanswers, True

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
Simple helper function that clears the previous status message and creates a new status message.
Status messages are important because the whole process takes much longer than users are used
to when using generic GPT
'''
def print_status(msg):
    print('\x1b[2K\r', end='')
    print(msg, end='\r')

'''
This is the core process of finding a section of text from the post that contains the answer,
generating a response to the user based on that text and then finally
performing the act of self-reflection that asks whether the answer generated is really 
supported by the quote. It is interesting how often GPT decides that its own 
answer is not actually supported by the text. This proves the value of self-reflection.
'''
async def create_response(nd_idxs_best, i_argmax, lpost_texts, question):
    idx_best_post = nd_idxs_best[i_argmax]
    print_status('extracting the relevant section...')

    role = ' '.join('Your job is to extract all sections from the text \
            in the \"Post\" section of the user content that would provide \
            an answer to the user\'s question in the \"Question\" section.'.split())
    quote = await chatgpt_req(role, [('user', f'Post:\n{lpost_texts[idx_best_post]}\nQuestion:\n{question}')])
    print_status('generating GPT response...')
    role = ' '.join('You are a helpful assistant. The text the appears \
            in the \"Background\" section is a selection from a post written by Eliyah23rd. \
            Please read it carefully and answer the \
            user\'s question in the \"Question\" section. The user\'s question is about what \
            Eliyah23rd says in his posts and not a general question so please try to give an answer \
            as specified in the text or clearly implied by it. \
            It is very important that you do not use the words \"Background\" or \"Question\" in your response'.split())
    response = await chatgpt_req(role, [('user', f'Background:\n{quote}\nQuestion:\n{question}')])
    print_status('verifying the response...')
    role = ' '.join('Your job is to determine whether the answer provided in \
            \"Answer\" section is supported by the text in the \"Background\" \
            section. For reference the original question appears in the \
            \"Question\" section. \
            Please provide only a number from 1 to 10 where 1 indicates that the question is not answered \
            at all in the post section and 10 indicates that the question is addressed directly.\n \
            It is very important that you only provide a number from 1 to 10 and no other words in your answer. '.split())
    score_str = await chatgpt_req(role, [('user', f'Background:\n{quote}\nQuestion:\n{question}\nAnswer:\n{response}')])
    try:
        score = int(score_str)
    except ValueError:
        score = -1
    if score < 4:
        '''
        In case we do fail, we want to generate a novel way of telling the user so rather than always 
        printing the same message as traditional software always does.
        Note. I raise the temperature a bit to avoid repetition.
        '''
        response = await chatgpt_req('Your job is to inform the user that you cannot answer the question.', temp=1.0)

    return response

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
            continue
        print('\n\n')
        print_status('creating embed for user input...')
        nd_qembed = get_embeds([question])[0]
        nd_scores = cosine_similarity(nd_qembed, nd_embeds)
        nd_idxs_best = nd_scores.argsort()[-c_num_closest:]
        print_status('checking with GPT which post is relevant ...')
        l_score_strs, b_extra_validation = await verify_relevant(nd_scores, nd_idxs_best, lpost_texts, question)
        nd_relevance_scores = np.zeros(len(l_score_strs))
        for iscore, score_str in enumerate(l_score_strs):
            try:
                nd_relevance_scores[iscore] = float(score_str)
            except ValueError:
                nd_relevance_scores[iscore] = -1
        i_argmax = np.argmax(nd_relevance_scores)
        if nd_relevance_scores[i_argmax] < 4:
            '''
            Note again the increase in temperature.
            '''
            response = await chatgpt_req('Your job is to inform the user that none of Eliyah\'s posts answer the question.', temp=1.0)
            print(response)
        else:
            response = await create_response(nd_idxs_best, i_argmax, lpost_texts, question)
        print(response)
        '''
        Here is the section where we get some feedback from the user.
        We've just given them the answer and ask them from some freeform text.
        We choose to translate this freeform answer into a number from 1 to 10
        Of course we use GPT to do this sentiment evaluation.
        '''
        feedback_question = 'Please tell me your whether you are satisfied with this answer.'
        user_feedback = input(feedback_question + '\n')
        role = ' '.join('Your role is to evaluate user satisfaction. \
                The question that we asked the user is found in the \"Question\" section and the user\'s \
                response is found in the "Response" section. \n \
                On a scale of 1 to 10 how would you rate the user\'s satisfaction? \n \
                Please provide only a number from 1 to 10 where 1 indicates that the user is very dissatisfied \
                with our answer and 10 indicates that the user is extremely satisfied with our answer. \
                It is very important that you only provide a number from 1 to 10 and no other words in your answer.\
                I must repeat that you must provide onle a single digit in your answer and no other text and \
                if you fail to follow this instruction, the answer will disrupt future operation of this program'.split())
        feedback_score_str = await chatgpt_req(role,
                [('user', f'\Question:\n{feedback_question}\nResponse:\n{user_feedback}')])
        try:
            feedback_score = int(feedback_score_str)
        except ValueError:
            feedback_score = -1
        
        extra_validation_response = 'yes' if b_extra_validation else 'no'
        extra_history = f'For the question \"{question}\" you answered {extra_validation_response} and the user \
        satisfaction was {feedback_score} out of 10.\n'
        extra_history = ' '.join(extra_history.split())
        with open('extra_history.txt', 'at') as fh_extra_history:
            fh_extra_history.write(extra_history + '\n')

            

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


