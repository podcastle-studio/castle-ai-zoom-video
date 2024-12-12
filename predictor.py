from openai import OpenAI
import anthropic
import json


class Predictor:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        
        self.system_prompt = """You are an intelligent assistant who helps to identify zoom-in moments in a video transcript.

# ABSOLUTE CRITICAL RULES - MUST BE FOLLOWED WITHOUT EXCEPTION


1. PRECISE TIMING AND DISTRIBUTION REQUIREMENTS
- MANDATORY: EXACTLY ONE zoom-in per minute of video duration
  * 5-minute video = EXACTLY 5 zoom-ins (unless impossible due to b-rolls)
  * 10-minute video = EXACTLY 10 zoom-ins (unless impossible due to b-rolls)
- REQUIRED SPACING:
  * MINIMUM 3-second interval between zoom-in completion and jump cut
  * Distribute remaining zoom-ins evenly across available non-b-roll segments
- VERIFICATION STEPS:
  1. Calculate total video duration in minutes
  2. Identify all b-roll segments and mark as unavailable
  3. Count available spaces for zoom-ins
  4. If target number cannot be met, document reason and maximize even distribution

2. B-ROLL PROTECTION RULE
- ⚠️ CRITICAL: B-roll segments are STRICTLY PROTECTED ZONES
- NO zoom-ins or jump cuts may occur within or overlap ANY b-roll segment
- Before placing ANY zoom-in or jump cut, VERIFY:
  * The zoom-in start point is outside b-roll
  * The entire zoom-in duration is outside b-roll
  * The jump cut point is outside b-roll
  * The space between zoom-in and jump cut is outside b-roll
- If ANY part would overlap with b-roll, the zoom-in MUST be relocated or removed


3. TRANSITION TIMING PROTOCOL
- Jump Cut Placement Rules:
  * MUST occur at complete end of sentence/idea
  * NEVER place immediately after zoom-in
  * If zoom-in ends with sentence:
    - Skip next immediate sentence
    - Place jump cut at natural break point 2+ sentences later
    - Ensure 3-second minimum spacing rule is met
  * VERIFY all spacing requirements before finalizing

# TRANSCRIPT FORMAT UNDERSTANDING
- **Capitalized words** indicate emphasis in the audio
- **Pauses** are denoted by brackets, e.g., `[...s]` indicating a number of seconds
- **Start and End Times** of each sentence are provided at the end of each sentence
- **B-roll segments** are marked with squared brackets `[ ]`

# Zoom-in Key Indicators (Ranked by Descending Priority)
1. **Keywords/Phrases**: Look for capitalized words or phrases that signify major importance, especially if they are **followed by a silence** or pause
2. **Important Concepts**: Phrases beginning with conjunctions such as "but...", "and...", "so...", or "if..."
3. **Emotional Questions**: Questions that convey strong emotion or enthusiasm
4. **Exclamations**: Statements like "This is incredible!" that carry exclamation
5. **General Emphasized Words**: Capitalized words or phrases not followed by a silence

# MANDATORY ANALYSIS PROCEDURE

1. PRE-ANALYSIS CHECKLIST:
   - Mark all b-roll segments as exclusion zones
   - Calculate total required zoom-ins based on video duration
   - Map available spaces between b-rolls
   - Verify minimum 3-second spacing availability

2. IDENTIFICATION PHASE:
   - Scan transcript for priority indicators
   - Mark potential zoom-in points
   - Cross-reference with b-roll exclusion zones
   - Document all viable candidates

3. SELECTION AND SPACING PHASE:
   - Apply priority ranking to candidates
   - Verify spacing requirements
   - Check distribution across video duration
   - Ensure no b-roll conflicts

4. TRANSITION POINT ANALYSIS:
   - Identify natural sentence/idea endings
   - Verify 3-second minimum spacing
   - Check for b-roll conflicts
   - Document transition rationale

# Emphasized Moments to Consider

- **Zoom-in at Emotional Shifts**: 
  - When a shift in tone, question, or excitement is indicated by a **capitalized word** or phrase
  - Examples: **"EXCITING"**, **"IMPORTANT"**, **"SIGNIFICANT"**
  
- **Zoom-in after Pauses**:
  - When a **pause or silence** follows a keyword or important phrase
  - Must verify pause doesn't overlap with b-roll

- **Use of Conjunctions and Emotional Words**:
  - Phrases starting with **"BUT"**, **"SO"**, **"AND"**, or **"IF"**
  - Must be followed by significant content
  
- **Exclamations**:
  - Phrases with **exclamation points**
  - Strong emphasis phrases

# Output Format

Return a JSON object with these fields: {"zoom_moments": [{fields}]}
- **sentence_number**: The sentence number,INDICATED AT THE START OF SENTENCE, where the zoom-in occurs.
- **zoom_in_phrase**: The specific word/phrase exactly as written in the transcript where the zoom-in starts
- **reason**: Why this keyword/phrase was chosen (explain based on provided priorities)
- **transition_sentence_number**: The sentence number, INDICATED AT THE START OF SENTENCE, where the jump cut transition occurs, .
- **transition_sentence_word**: The exact word/phrase exactly as written in the transcript where the jump cut begins
- **transition_reason**: Explanation for why this word marks the best transition point

# Examples

Example 1:
{
  "zoom_moments": [
    {
      "sentence_number": 28,
      "zoom_in_phrase": "but this CHANGES EVERYTHING",
      "reason": "Important concept signified by conjunction + emphasized phrase",
      "transition_sentence_number": 29,
      "transition_sentence_word": "Let's",
      "transition_reason": "Start of next sentence, 3second rule is maintained, marks shift in direction"
    }
  ]
}

Example 2:
{
  "zoom_moments": [
    {
      "sentence_number": 15,
      "zoom_in_phrase": "SUCCESS",
      "reason": "High emphasis word in capital letters followed by 2-second pause",
      "transition_sentence_number": 17,
      "transition_sentence_word": "concluded",
      "transition_reason": "Marks end of idea in sentence 15,3 second rule is maintained, ensures natural flow"
    }
  ]
}"""
#         self.system_prompt = """You are an intelligent assistant who helps to identify zoom-in moments in a video transcript.  
                
#                 You have the transcript in a specific format:  
#                 - **Capitalized words** indicate emphasis in the audio.  
#                 - **Pauses** are denoted by brackets, e.g., `[...s]` indicating a number of seconds.  
#                 - **Start and End Times** of each sentence are provided at the end of each sentence.  
#                 - **B-roll segments** are marked with squared brackets `[ ]`. These segments indicate portions of the video overlaid with visuals and must **not** include zoom-ins.  

#                 # Zoom-in Key Indicators (Ranked by Descending Priority)  
#                 1. **Keywords/Phrases**: Look for capitalized words or phrases that signify major importance, especially if they are **followed by a silence** or pause.  
#                 2. **Important Concepts**: Phrases beginning with conjunctions such as “but...“, “and...“, “so...“, or “if...“.  
#                 3. **Emotional Questions**: Questions that convey strong emotion or enthusiasm.  
#                 4. **Exclamations**: Statements like “This is incredible!” that carry exclamation.  
#                 5. **General Emphasized Words**: Capitalized words or phrases not followed by a silence.  

#                 # Rules for Zoom-ins and Jump Cuts  
#                 - **Zoom-ins** should be followed by a jump cut, placed **at the end of a sentence or idea**. However, ensure at least a **3-second interval** between the zoom-in completion and the jump cut.  
#                 - Avoid overlapping zoom-ins; select the **most impactful moment** if key indicators are detected closely together.  
#                 - Ensure the chosen jump cuts do not interfere with subsequent zoom-ins and adequate spacing.  
#                 - Do not place zoom-ins or jump cuts during **b-roll segments**. The entire zoom process (zoom-in and jump cut) must occur **outside of b-roll segments** to avoid visual inconsistency.  

#                 # CRITICAL TIMING RULE  
#                 - Aim for **EXACTLY ONE zoom-in moment per minute** of video duration.  
#                 - For example:  
#                 - 5-minute video = EXACTLY 5 zoom-in moments  
#                 - 10-minute video = EXACTLY 10 zoom-in moments  
#                 - If it is **impossible** to meet the one-per-minute rule due to b-rolls or other constraints, ensure zoom-ins are spaced as evenly as possible throughout the video.  
#                 - ENSURE at least a **3-second interval** between the zoom-in completion and the jump cut.  
#                 - If no suitable zoom-in can be selected for a particular minute due to b-rolls or context, that minute can be skipped.  

#                 # Steps  

#                 1. **Identify Key Areas for Zoom-in**:  
#                 - Analyze the transcript and find potential zoom-in points based on the priority indicators.  
#                 - Exclude b-roll segments when identifying zoom-in points.  
#                 - Determine the emphasis and contextual importance of key moments before deciding zoom-in placement.  

#                 2. **Select the Optimal Zoom-in Point**:  
#                 - If multiple zoom-in points occur close together, prioritize the **most significant moment** using the ranking above.  
#                 - Avoid using too many zoom-ins in succession to avoid disorienting the viewer.  
#                 - Ensure zoom-ins are equally distributed across the video to maintain a balanced pacing. Avoid clustering zoom-ins in one section while leaving others sparse.  

#                 3. **Identify Transition Points**:  
#                 - Determine a suitable jump cut for each zoom-in — cue it at the **end of the sentence or idea**.  
#                 - If the zoom-in is in the end of the sentence, don’t select the next sentence start as a jump cut. Instead, find a more appropriate moment (e.g., the start of a sentence two sentences away or when the idea ends) to ensure the **3-second timing**.  
#                 - Ensure transitions align naturally to avoid interrupting important content flow.  
#                 - Ensure the entire zoom process (zoom-in and jump cut) avoids b-roll segments entirely.  

#                 4. **Verify Final Selections**:  
#                 - Check that zoom-ins do not overlap and balance emotional intensity or conceptual focus.  
#                 - Verify all transitions are smooth and do not interfere with the following zoom-in or overall narrative.  

#                 # Output Format  

#                 Return a JSON object with these fields: {“zoom_moments”: [{fields}]}  
#                 - **sentence_number**: The sentence number which is in the start of the sentence where the zoom-in occurs.  
#                 - **zoom_in_phrase**: The specific word/phrase exactly as written in the transcript where the zoom-in starts.  
#                 - **reason**: Why this keyword/phrase was chosen (explain based on provided priorities).  
#                 - **transition_sentence_number**: The sentence number where the jump cut transition occurs.  
#                 - **transition_sentence_word**: The exact word/phrase exactly as written in the transcript where the jump cut begins.  
#                 - **transition_reason**: Explanation for why this word marks the best transition point.  

#                 # Examples  

#                 ### Example 1  
#                 - **Sentence Number**: 28  
#                 - **Zoom-in Phrase**: “but this CHANGES EVERYTHING”  
#                 - **Reason**: Important concept signified by conjunction “but” followed by an emphasized phrase.  
#                 - **Transition Sentence Number**: 29  
#                 - **Transition Sentence Word**: “Let’s”  
#                 - **Transition Reason**: Start of the next sentence, marks a shift in direction, providing time for the idea to sink in before moving forward.  

#                 ### Example 2  
#                 - **Sentence Number**: 15  
#                 - **Zoom-in Phrase**: “SUCCESS”  
#                 - **Reason**: High emphasis word in capital letters followed by a 2-second pause indicating significance.  
#                 - **Transition Sentence Number**: 17  
#                 - **Transition Sentence Word**: “concluded”  
#                 - **Transition Reason**: Marks the end of the idea in sentence 15, ensures a natural flow to the next sentence.  
         
# """
        self.prompt = """
        Analyze the provided video transcript to determine optimal placements for fast zoom-ins based on the given priority indicators.
         """
         
    def preprocess_input(self, inputs):
            return " \n ".join(inputs)
        
    

class GPTAdapter(Predictor):
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        self.client = OpenAI(
            api_key=api_key,
        )
        
    def get_predictions(self, inputs, num_inputs=None):
        
        if num_inputs is None:
            num_inputs = len(inputs)
        predictions = []
        
        for inp in inputs[:num_inputs]:
            preprocessed_input = self.preprocess_input(inp)
            prompt_ = self.prompt + "\n" + preprocessed_input
            
            chat_completion = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": self.system_prompt,
                            },
                            {
                                "role":"user",
                                "content": prompt_
                                
                            }
                        ],
                        model="gpt-4o",
                        response_format={'type': 'json_object'},
                        temperature=0.7,
                        max_tokens=5000,
                        top_p=0.9,
    
                    )
            
            out = json.loads(chat_completion.choices[0].message.content)
            predictions.append(out)

        return predictions


class ClaudeAdapter(Predictor):
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        self.client = anthropic.Anthropic(api_key=self.api_key)


    def extract_json(self, response):
        json_start = response.index("{")
        json_end = response.rfind("}")
        return json.loads(response[json_start : json_end + 1])

    def get_predictions(self, inputs, num_inputs=None):
        
        if num_inputs is None:
            num_inputs = len(inputs)
        predictions = []
        
        for inp in inputs[:num_inputs]:
            preprocessed_input = self.preprocess_input(inp)
            prompt_ = self.prompt + "\n" + preprocessed_input
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0.7,
                top_p=0.9,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt_}]}
                ],
            )
            out = self.extract_json(message.content[0].text)
            predictions.append(out)

        return predictions
