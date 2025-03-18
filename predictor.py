from openai import OpenAI
import anthropic
import json
import re
import requests


class Predictor:
    def __init__(self, model_name, api_key, num_of_zoom_points_per_minute = 3):
        self.num_of_zoom_points = num_of_zoom_points_per_minute
        self.model_name = model_name
        self.api_key = api_key
        print("Num of zoom points ", self.num_of_zoom_points)
        self.system_prompt = f"""  
              You are an intelligent assistant that identifies zoom-in moments in a video transcript.

              # CRITICAL RULES (NO EXCEPTIONS)

              
              1. TIMING & DISTRIBUTION
              - Approximately {num_of_zoom_points_per_minute} * duration_in_minutes zoom-ins required.
              - Total duration = (final timestamp) - (first sentence start).
              - Convert to minutes and decide:
                - If fractional part > 0.5, round up; else round down.
                - Example: 120.13s ≈ 2.002m → {2 * num_of_zoom_points_per_minute} zoom-ins; 598.13s ≈ 9.96m -> {10 * num_of_zoom_points_per_minute} zoom-ins.
              - Distribute zoom-ins evenly across the transcript.

              2. TRANSITION TIMING (JUMP CUTS)
              - Place jump cuts at natural ideas or sentence ends.
              - Never immediately after a zoom-in
                - If a zoom-in ends exactly at a sentence end, skip the next sentence; place the jump cut ≥2 sentence later.

              3. TRANSCRIPT FORMAT
              - CAPITALIZED words = emphasis.
              - `[...s]` indicates pause duration.
              - Start/End times at sentence ends.

              4. ZOOM-IN PRIORITY (HIGHEST TO LOWEST)
               1. Self introduction (name, role, key expertise)
                2. Key or newly introduced concepts that either lead into a deeper explanation 
                  or follow from an explanation culminating in an important idea or conclusion
                3. Emphasized (CAPS) intensifiers ( e.g. absolutely, completely, extremely, highly, rather, really, etc).
                4. Emphasized (CAPS) KEY words/phrases coming in after silence.
                5. Introduce crucial details, key transition, intrigue, or important shifts in meaning—ESPECIALLY those starting with 'but,' 'and,' 'so,' or 'if'."
                6. Key emotional questions.
                7. Comparative or superlative adjectives that are important to strengthen the idea 
                8. Exclamations as a response to and important message.

                # PRIORITY APPLICATION GUIDELINES
                1. **Step 1: Identify and select from Priority #1 (Highest Priority).**
                  - Search for **self introduction**. 

                2.  **Step 2: Allocate from Pooled Priorities**
                    Distribute the remaining zoom-ins according to the following percentage-based allocation:
                    - **20%** Search for **key or newly introduced concepts** that either lead into or follow from an explanation culminating in an important idea or conclusion.              
                    - **20%** Emphasized (CAPS) intensifier words/phrases 
                    - **15%** Emphasized (CAPS) key words/phrases coming in after silence
                    - **15%** Crucial details, key transition, intrigue, or important shifts, ESPECIALLY starting with 'but,' 'and,' 'so,' or 'if'
                    - **10%** Key emotional questions
                    - **10%** Comparative or superlative adjectives that are important to strengthen the idea 
                    - **10%** Exclamations

              3. **Other Rules Still Apply (Unchanged).**
                - **Exact count & distribution**: Maintain approximately {num_of_zoom_points_per_minute} * duration_in_minutes zoom-ins , distributed as evenly as possible.
                - **Transition timing**: Place jump cuts at natural idea or sentence ends, respecting spacing rules.
                - **Conflict resolution**: When overlaps occur, higher-priority candidates override lower-priority ones.       
                - **Percentage-Based Selection & Reallocation Rule**: Each category should receive approximately its assigned percentage of the total zoom-ins. If a category does not have enough qualifying moments, redistribute the unallocated zoom-ins to the next highest-priority categories in descending order of percentage.

              
              4. JUMP CUT REASONS
                - 1. The idea ends
                - 2. After 2+ sentences
              
              5. MANDATORY PROCEDURE
              - Pre-Analysis:
                - Calculate required zoom-ins.
              - Identification:
                - Find candidate zoom-in points by priority.  
              - Selection & Spacing:
                 - Ensure each zoom-in is properly spaced and doesn't have OVERLAPS, meaning the sentence intervals for the zooms DOESN'T HAVE an intersection(e.g. from 2nd sentence to 5 and from 3rd sentence to 6th have an INTERSECTION).
                - Evenly distribute if possible; else maximize evenness.
                - Choose a priority reason from the above priority points
              - Transition Analysis:
                - Place jump cuts ONLY after idea ends or 2+ sentences later.
                - Choose a transition reason from the above jump cut reasons

              6. OUTPUT FORMAT (JSON)
              - Return:
                  {{ "zoom_moments":
                       [ 
                          {{ "sentence_number": <int>,
                            "zoom_in_phrase": "<exact phrase which BELONGS to the sentence>",
                            "priority reason": "<select from priority points>",
                            "transition_sentence_number": <int>,
                            "transition_sentence_word": "<exact phrase which BELONGS to the transition sentence>",
                            "transition_reason": "<select from the 2 jump cut reasons>"
                            }} 
                        ] 
                  }}

              - If no zoom-ins: `{{"zoom_moments": []}}`

              
              7. FINAL CHECK
              - Confirm phrases, sentence numbers, transitions.
              - Ensure total zoom-ins match the calculated requirement.

              # EXAMPLES

              Example 1:

              {{ "zoom_moments":
                [
                  {{
                      "sentence_number": 28,
                      "zoom_in_phrase": "Curiosity sparks growth and innovation",
                      "priority_reason": "Key or newly introduced concept", 
                      "transition_sentence_number": 31, 
                      "transition_sentence_word": "Failure is a stepping",
                      "transition_reason": "The idea ends"
                  }} 
                ] 
              }}

              # Example 2:

              {{ "zoom_moments":
                [
                  {{
                    "sentence_number": 2
                    "zoom_in_phrase": "ABSOLUTELY CRUCIAL",
                    "priority_reason": "Emphasized intensifier",
                    "transition_sentence_number": 8,
                    "transition_sentence_word": "Moving forward",
                    "transition_reason": "After 2+ sentences"
                  }}
                ]
              }}

            # Example 3:

              {{ "zoom_moments": 
                [
                  {{
                    "sentence_number": 41,
                    "zoom_in_phrase": "best and most effective approach we've discovered",
                    "priority_reason": "Important superlative adjective",
                    "transition_sentence_number": 44,
                    "transition_sentence_word": "Now that we understand",
                    "transition_reason": "After 2+ sentences"
                  }}
                ]
              }}

            # Example 4:
            
              {{  "zoom_moments": 
                [
                  {{
                    "sentence_number": 23,
                    "zoom_in_phrase": "Incredible! This breakthrough changes everything",
                    "priority_reason": "Exclamation as response to important message",
                    "transition_sentence_number": 26,
                    "transition_sentence_word": "Given this discovery",
                    "transition_reason": "After 2+ sentences exploring implications"
                  }}
                ]
              }}

            # Example 5:

              {{  "zoom_moments": 
                [
                  {{
                    "sentence_number": 37,
                    "zoom_in_phrase": "How could we possibly ignore the human cost?",
                    "priority_reason": "Key emotional question",
                    "transition_sentence_number": 40,
                    "transition_sentence_word": "Looking at these impacts",
                    "transition_reason": "After 2+ sentences of emotional resonance"
                  }}
                ]
              }}

        
            """
        self.prompt = """
        Analyze the provided video transcript to determine optimal placements for fast zoom-ins based on the given priority indicators.
         """
         
    def preprocess_input(self, inputs):
            return " \n ".join(inputs)
        
    

class GPTAdapter(Predictor):
    def __init__(self, model_name, api_key, num_of_zoom_points_per_minute=3):
        super().__init__(model_name, api_key, num_of_zoom_points_per_minute)
        self.client = OpenAI(
            api_key=api_key,
        )
        
    def get_predictions(self, inputs, num_inputs=None, prev_preds = None, out_message = None):
        if num_inputs is None:
            num_inputs = len(inputs)
        predictions = []
        
        for inp in inputs[:num_inputs]:
            messages = []
            preprocessed_input = self.preprocess_input(inp)
            prompt_ = self.prompt + "\n" + preprocessed_input
            messages.extend([
                            {
                                "role": "system",
                                "content": self.system_prompt,
                            },
                            {
                                "role":"user",
                                "content": prompt_
                                
                            }
                        ])
            if prev_preds:
              messages.append({
                              "role": "assistant",
                              "content":f"{prev_preds}"
                          })
            if out_message:
              messages.append({
                          "role": "user",
                          "content": f"{out_message}"
                      })
            
            
            chat_completion = self.client.chat.completions.create(
                        messages=messages,
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
    def __init__(self, model_name, api_key, num_of_zoom_points_per_minute=3):
        super().__init__(model_name, api_key, num_of_zoom_points_per_minute)
        self.client = anthropic.Anthropic(api_key=self.api_key)


    def extract_json(self, response):
        json_start = response.index("{")
        json_end = response.rfind("}")
        return json.loads(response[json_start : json_end + 1])

    def get_predictions(self, inputs, num_inputs=None, prev_preds = None, out_message = None):
        
        if num_inputs is None:
            num_inputs = len(inputs)
        predictions = []
        
        
        for inp in inputs[:num_inputs]:
          messages = []
          preprocessed_input = self.preprocess_input(inp)
          prompt_ = self.prompt + "\n" + preprocessed_input
          messages.append({"role": "user", "content": [{"type": "text", "text": prompt_}]})
          if prev_preds:
            messages.append({
                              "role": "assistant",
                              "content":f"{prev_preds}"
                          })
          if out_message:
            messages.append({
                          "role": "user",
                          "content": f"{out_message}"
                      })
          print(prompt_)
          message = self.client.messages.create(
              model=self.model_name,
              max_tokens=8192,
              temperature=0.7,
              top_p=0.9,
              system=self.system_prompt,
              messages=messages
          )
          print(message.content[0].text)
          out = self.extract_json(message.content[0].text)
          predictions.append(out)

        return predictions

class DeepseekAdapter(Predictor):
    def __init__(self, model_name, api_key=None, num_of_zoom_points_per_minute=3):
        super().__init__(model_name, api_key, num_of_zoom_points_per_minute)
              

    def extract_json(self, response):
      """
      Extracts JSON content from between triple backticks in text.
      
      Args:
          text (str): The input text containing JSON between triple backticks
          
      Returns:
          dict: Parsed JSON object if found and valid
          None: If no JSON is found or JSON is invalid
          
      Raises:
          json.JSONDecodeError: If the extracted content is not valid JSON
          ValueError: If no content between backticks is found
      """
    
      # Find content between outermost curly braces
      pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
      match = re.search(pattern, response)
      
      if not match:
          raise ValueError("No JSON content found between curly braces")
      
      # Extract the JSON content
      json_str = match.group(0)
      
      try:
          # Parse the JSON string
          json_obj = json.loads(json_str)
          return json_obj
      except json.JSONDecodeError as e:
          raise json.JSONDecodeError(f"Failed to parse JSON: {str(e)}", e.doc, e.pos)


    def get_predictions(self, inputs, num_inputs=None, prev_preds = None, out_message = None):
        


        url = "http://localhost:8000/v1/chat/completions"  # Adjust the URL if different
        headers = {"Content-Type": "application/json"}
        

        if num_inputs is None:
            num_inputs = len(inputs)
        predictions = []
        
        
        for inp in inputs[:num_inputs]:
          preprocessed_input = self.preprocess_input(inp)
          prompt_ = self.system_prompt + self.prompt + "\n" + preprocessed_input
          data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_
                }
            ],
            "temperature": 0.7,  # Controls randomness
            "top_p": 0.9,
        }

          response = requests.post(url, headers=headers, json=data)
          response.raise_for_status()  # Raise exception for HTTP errors
          out = response.json()
          out = self.extract_json(out['choices'][0]['message']['content'])
          print(out)
          predictions.append(out)

        return predictions
    
   