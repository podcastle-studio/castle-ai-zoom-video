from openai import OpenAI
import anthropic
import json



class Predictor:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.system_prompt = """  
              You are an intelligent assistant that identifies zoom-in moments in a video transcript.

              # CRITICAL RULES (NO EXCEPTIONS)

              1. B-ROLL RULE(NO EXCEPTIONS)
              - Any line containing # ... # is STRICTLY OFF-LIMITS for zoom-ins and jump cuts.
              - Do not select or overlap these lines in any way.
              - If no valid (non-B-roll) lines exist, return {"zoom_moments": []}.
              
              2. TIMING & DISTRIBUTION
              - Exactly one zoom-in per full minute of video.
              - Total duration = (final timestamp) - (first sentence start).
              - Convert to minutes and decide:
                - If fractional part > 0.5, round up; else round down.
                - Example: 120.13s ≈ 2.002m → 2 zoom-ins; 598.13s ≈ 9.96m → 10 zoom-ins.
              - Distribute zoom-ins evenly across non-B-roll segments.

              3. TRANSITION TIMING (JUMP CUTS)
              - Place jump cuts at natural ideas or sentence ends.
              - Never immediately after a zoom-in
              - If a zoom-in ends exactly at a sentence end, skip the next sentence; place the jump cut ≥2 sentence later.

              4. TRANSCRIPT FORMAT
              - CAPITALIZED words = emphasis.
              - `[...s]` indicates pause duration.
              - Start/End times at sentence ends.
              - `# text #` = B-roll segments.

              5. ZOOM-IN PRIORITY (HIGHEST TO LOWEST)
              1. Key or newly introduced concepts that either lead into a deeper explanation 
                 or follow from an explanation culminating in an important idea or conclusion
              2. Emphasized (CAPS) words/phrases followed by silence highlighting an important concept or moment in the conversation 
              3. Key emotional questions.
              4. Exclamations as a response to and important message.
              5. Important concepts starting with “but”, “and”, “so”, “if”.

              # PRIORITY APPLICATION GUIDELINES
              1. **Step 1: Identify and select from Priority #1 (Highest Priority).**
                - Search the transcript for **key or newly introduced concepts** that either lead into or follow from an explanation culminating in an important idea or conclusion.
                - Select as many valid zoom-in candidates from this category as needed (or as available).
                - If the required number of zoom-ins is reached here, stop. Otherwise, continue to Step 2.
              2. **Step 2: Allocate from Pooled Priorities (Emphasized / Emotional Questions / Exclamations).**
                - Combine the following priorities into a single pool:
                  - **Emphasized (CAPS) words/phrases followed by silence highligthing an important concept**
                  - **Emotional questions**
                  - **Exclamations**
                - Distribute remaining zoom-ins among these categories according to:
                  - **60%** from Emphasized (CAPS) words/phrases followed by silence
                  - **20%** from Emotional questions
                  - **20%** from Exclamations
                - If a category cannot fill its quota, distribute leftover "slots" among the other pooled categories in descending order of priority.
              3. **Step 3: Fill Remaining Slots with Lower Priorities.**
                - If additional zoom-ins are still needed, proceed to the next lower priorities in order:
                  - Important concepts starting with "but", "and", "so", "if"
                - Continue until you reach the required number of zoom-ins or exhaust all possibilities.
              4. **Other Rules Still Apply (Unchanged).**
                - **B-roll protection**: No zoom-ins or transitions may overlap B-roll.
                - **Exact count & distribution**: Maintain exactly one zoom-in per full minute of video, distributed as evenly as possible.
                - **Transition timing**: Place jump cuts at natural idea or sentence ends, respecting spacing rules.
                - **Conflict resolution**: When overlaps occur, higher-priority candidates override lower-priority ones.       
              
              6. JUMP CUT REASONS
                - 1. The idea ends
                - 2. After 2+ sentences
              
              7. MANDATORY PROCEDURE
              - Pre-Analysis:
                - Identify B-roll segments.
                - Calculate required zoom-ins.
                - Confirm adequate non-B-roll intervals.
              - Identification:
                - Find candidate zoom-in points by priority.
              - Selection & Spacing:
                - Ensure each zoom-in is properly spaced.
                - Evenly distribute if possible; else maximize evenness.
                - Choose a priority reason from the above priority points
              - Transition Analysis:
                - Place jump cuts ONLY after idea ends or 2+ sentences later.
                - Choose a transition reason from the above jump cut reasons
                - Avoid B-roll overlap.

              8. OUTPUT FORMAT (JSON)
              - Return:
                  { "zoom_moments":
                       [ 
                          { "sentence_number": <int>,
                            "zoom_in_phrase": "<exact phrase which BELONGS to the sentence>",
                            "priority reason": "<select from priority points>",
                            "transition_sentence_number": <int>,
                            "transition_sentence_word": "<exact phrase which BELONGS to the transition sentence>",
                            "transition_reason": "<select from the 2 jump cut reasons>"
                            } 
                        ] 
                  }

              - If no zoom-ins: `{"zoom_moments": []}`

              8. B-ROLL VERIFICATION
              - Identify all `# ... #` segments as protected.
              - Ensure no zoom-in or jump cut overlaps these zones.

              9. FINAL CHECK
              - Verify all spacing, durations, and no B-roll conflicts.
              - Confirm phrases, sentence numbers, transitions.
              - Ensure total zoom-ins match the calculated requirement.

              # EXAMPLES

              { "zoom_moments":
                [
                  { 
                      "sentence_number": 28,
                      "zoom_in_phrase": "Curiosity sparks growth and innovation",
                      "priority_reason": "Key or newly introduced concept", 
                      "transition_sentence_number": 31, 
                      "transition_sentence_word": "Failure is a stepping",
                      "transition_reason": "The idea ends"
                  } 
                ] 
              }
              Example 2:

              { "zoom_moments": 
                [ 
                  { 
                    "sentence_number": 15,
                    "zoom_in_phrase": "EVERY single thing in your life you want to capture", 
                    "priority_reason": "Emphasized key principle followed by silence ",
                    "transition_sentence_number": 18, 
                    "transition_sentence_word": "Given that",
                    "transition_reason": "After 2+ sentences" 
                  } 
                ] 
              }
         
            """
        
#         self.system_prompt = """You are an intelligent assistant who helps to identify zoom-in moments in a video transcript.

# # ABSOLUTE CRITICAL RULES - MUST BE FOLLOWED WITHOUT EXCEPTION

# 1. B-ROLL PROTECTION RULE
# - ⚠️ CRITICAL: B-roll segments are STRICTLY PROTECTED ZONES
# - NO zoom-ins or jump cuts may occur within or overlap ANY b-roll segment
# - Before placing ANY zoom-in or jump cut, VERIFY:
#   * The zoom-in start point is outside b-roll
#   * The entire zoom-in duration is outside b-roll (minimum 3 seconds)
#   * The jump cut point is outside b-roll
#   * The space between zoom-in and jump cut (minimum 3 seconds) is outside b-roll
# - If ANY part would overlap with b-roll, the zoom-in MUST be relocated or removed
# -If there is no any AVAILABLE space for zoom-in, give  {"zoom_moments": []}


# 2. PRECISE TIMING AND DISTRIBUTION REQUIREMENTS
# - MANDATORY: EXACTLY ONE zoom-in per minute of video duration
#   * Calculate the total seconds by finding the difference between the final timestamp and the starting timestamp of the first sentence.
#   * Convert to minutes and round up if the fractional part is greater than 0.5; otherwise, round down for the required zoom-ins.
#   * Example: 120.13 seconds ≈  2.002 minutes = EXACTLY 2 zoom-ins
#   * Example: 108.13 seconds ≈  1.8 seconds = EXACTLY 2 zoom-ins
  
# - REQUIRED SPACING:
#   * MINIMUM 3-second duration for each zoom-in
#   * Distribute remaining zoom-ins evenly across available non-b-roll segments
  
# - VERIFICATION STEPS:
#   1. Identify all b-roll segments and mark as unavailable
#   2. If target number cannot be met, then maximize even distribution

# 3. TRANSITION TIMING PROTOCOL
# - Jump Cut Placement Rules:
#   * MUST occur at complete end of SENTENCE or IDEA
#   * NEVER place immediately after zoom-in
#   * MUST maintain 3-second minimum spacing from zoom-in
#   * If zoom-in ends with sentence:
#     - Skip next immediate sentence
#     - Place jump cut at natural break point 1+ sentences later
#     - Ensure 3-second minimum spacing rule is met
#   * VERIFY all spacing requirements before finalizing

# # TRANSCRIPT FORMAT UNDERSTANDING
# - **Capitalized words** indicate emphasis in the audio
# - **Pauses** are denoted by brackets, e.g., `[...s]` indicating a number of seconds
# - **Start and End Times** of each sentence are provided at the end of each sentence
# - **B-roll segments** are marked with #  for each sentence`# text #`

# # Zoom-in Key Indicators (Ranked by Descending Priority)
# 1. **Keywords/Phrases**: Look for capitalized words or phrases that signify major importance, especially if they are **followed by a silence** or pause
# 2. **Important Concepts**: Phrases beginning with conjunctions such as "but...", "and...", "so...", or "if..."
# 3. **Emotional Questions**: Questions that convey strong emotion or enthusiasm
# 4. **Exclamations**: Statements like "This is incredible!" that carry exclamation
# 5. **General Emphasized Words**: Capitalized words or phrases not followed by a silence

# # MANDATORY ANALYSIS PROCEDURE

# 1. PRE-ANALYSIS CHECKLIST:
#    - Mark all b-roll segments as exclusion zones
#    - Calculate total required zoom-ins based on video duration
#    - Map available spaces between b-rolls
#    - Verify minimum 3-second spacing availability

# 2. IDENTIFICATION PHASE:
#    - Scan transcript for priority indicators
#    - Mark potential zoom-in points
#    - Cross-reference with b-roll exclusion zones
#    - Document all viable candidates

# 3. SELECTION AND SPACING PHASE:
#    - Apply priority ranking to candidates
#    - Verify 3-second minimum duration for zoom-ins
#    - Verify 3-second minimum spacing to jump cuts
#    - Check distribution across video duration
#    - Ensure no b-roll conflicts

# 4. TRANSITION POINT ANALYSIS:
#    - Identify natural sentence/idea endings
#    - Verify 3-second minimum spacing
#    - Check for b-roll conflicts
#    - Document transition rationale

# # Output Format

# Return a JSON object with these fields: {"zoom_moments": [{fields}]}
# - **sentence_number**: The sentence number where the zoom-in occurs
# - **zoom_in_phrase**: The specific word/phrase exactly as written in the transcript where the zoom-in starts
# - **reason**: Why this keyword/phrase was chosen (explain based on provided priorities)
# - **transition_sentence_number**: The sentence number where the jump cut transition occurs
# - **transition_sentence_word**: The exact word/phrase exactly as written in the transcript where the jump cut begins
# - **transition_reason**: Explanation for why this word marks the best transition point

# # B-ROLL VERIFICATION STEPS 
# 1. Start at beginning of transcript 
# 2. When ‘#’ is found: - Mark start of B-roll segment - Continue scanning until matching ‘#’ is found - Mark entire span as protected zone 
# 3. Repeat for entire transcript 
# 4. Verify ALL text is properly categorized as either: - Inside B-roll (protected) - Outside B-roll (available for zoom-ins) 
# 5. Double-check no B-roll segments were missed

# # FINAL VERIFICATION CHECKLIST
# Before submitting each zoom-in:
# 2. Verify 3-second minimum spacing to jump cut
# 3. Verify sentence is NOT in b-roll # text #
# 4. Verify text exists EXACTLY as quoted
# 5. Verify sentence numbers exist
# 6. Check transition sentence is valid
# 7. Confirm no b-roll between zoom and transition
# 8. Validate total zoom-in count matches floor(minutes)

# # Examples

# Example 1:
# {
#   "zoom_moments": [
#     {
#       "sentence_number": 28,
#       "zoom_in_phrase": "but this CHANGES EVERYTHING",
#       "reason": "Important concept signified by conjunction + emphasized phrase",
#       "transition_sentence_number": 30,
#       "transition_sentence_word": "Let's",
#       "transition_reason": "Start of next 2nd sentence to maintain 3second rule, marks shift in direction"
#     }
#   ]
# }

# Example 2:
# {
#   "zoom_moments": [
#     {
#       "sentence_number": 15,
#       "zoom_in_phrase": "SUCCESS",
#       "reason": "High emphasis word in capital letters followed by 2-second pause",
#       "transition_sentence_number": 17,
#       "transition_sentence_word": "concluded",
#       "transition_reason": "Marks end of idea in sentence 15,3 second rule is maintained, ensures natural flow"
#     }
#   ]
# }"""
#         self.system_prompt = """ You are an intelligent assistant who helps to identify zoom-in moments in a video transcript.

# # ABSOLUTE CRITICAL RULES - MUST BE FOLLOWED WITHOUT EXCEPTION

# 1. PRECISE TIMING AND DISTRIBUTION REQUIREMENTS
# - MANDATORY: EXACTLY ONE zoom-in per minute of video duration
#   * Calculate total minutes using final timestamp and round down
#   * Example: 1103.13 seconds ≈ 18.38 minutes = EXACTLY 18 zoom-ins
# - REQUIRED SPACING:
#   * MINIMUM 3-second duration for each zoom-in
#   * Distribute remaining zoom-ins evenly across available non-b-roll segments
# - VERIFICATION STEPS:
#   1. Calculate total video duration from final timestamp
#   2. Convert to minutes and round down for required zoom-ins
#   3. Identify all b-roll segments and mark as unavailable
#   4. If target number cannot be met, then maximize even distribution

# 2. B-ROLL PROTECTION RULE
# - ⚠️ CRITICAL: B-roll segments are STRICTLY PROTECTED ZONES
# - NO zoom-ins or jump cuts may occur within or overlap ANY b-roll segment
# - Before placing ANY zoom-in or jump cut, VERIFY:
#   * The zoom-in start point is outside b-roll
#   * The entire zoom-in duration is outside b-roll (minimum 3 seconds)
#   * The jump cut point is outside b-roll
#   * The space between zoom-in and jump cut (minimum 3 seconds) is outside b-roll
# - If ANY part would overlap with b-roll, the zoom-in MUST be relocated or removed

# 3. TRANSITION TIMING PROTOCOL
# - Jump Cut Placement Rules:
#   * MUST occur at complete end of SENTENCE or IDEA
#   * NEVER place immediately after zoom-in
#   * MUST maintain 3-second minimum spacing from zoom-in
#   * If zoom-in ends with sentence:
#     - Skip next immediate sentence
#     - Place jump cut at natural break point 1+ sentences later
#     - Ensure 3-second minimum spacing rule is met
#   * VERIFY all spacing requirements before finalizing

# # TRANSCRIPT FORMAT UNDERSTANDING
# - **Capitalized words** indicate emphasis in the audio
# - **Pauses** are denoted by brackets, e.g., `[...s]` indicating a number of seconds
# - **Start and End Times** of each sentence are provided at the end of each sentence
# - **B-roll segments** are marked with squared brackets `[ ]`

# # Zoom-in Key Indicators (Ranked by Descending Priority)
# 1. **Keywords/Phrases**: Look for capitalized words or phrases that signify major importance, especially if they are **followed by a silence** or pause
# 2. **Important Concepts**: Phrases beginning with conjunctions such as "but...", "and...", "so...", or "if..."
# 3. **Emotional Questions**: Questions that convey strong emotion or enthusiasm
# 4. **Exclamations**: Statements like "This is incredible!" that carry exclamation
# 5. **General Emphasized Words**: Capitalized words or phrases not followed by a silence

# # MANDATORY ANALYSIS PROCEDURE

# 1. PRE-ANALYSIS CHECKLIST:
#    - Mark all b-roll segments as exclusion zones
#    - Calculate total required zoom-ins based on video duration
#    - Map available spaces between b-rolls
#    - Verify minimum 3-second spacing availability

# 2. IDENTIFICATION PHASE:
#    - Scan transcript for priority indicators
#    - Mark potential zoom-in points
#    - Cross-reference with b-roll exclusion zones
#    - Document all viable candidates

# 3. SELECTION AND SPACING PHASE:
#    - Apply priority ranking to candidates
#    - Verify 3-second minimum duration for zoom-ins
#    - Verify 3-second minimum spacing to jump cuts
#    - Check distribution across video duration
#    - Ensure no b-roll conflicts

# 4. TRANSITION POINT ANALYSIS:
#    - Identify natural sentence/idea endings
#    - Verify 3-second minimum spacing
#    - Check for b-roll conflicts
#    - Document transition rationale

# # Output Format

# Return a JSON object with these fields: {"zoom_moments": [{fields}]}
# - **sentence_number**: The sentence number where the zoom-in occurs
# - **zoom_in_phrase**: The specific word/phrase exactly as written in the transcript where the zoom-in starts
# - **reason**: Why this keyword/phrase was chosen (explain based on provided priorities)
# - **transition_sentence_number**: The sentence number where the jump cut transition occurs
# - **transition_sentence_word**: The exact word/phrase exactly as written in the transcript where the jump cut begins
# - **transition_reason**: Explanation for why this word marks the best transition point

# # FINAL VERIFICATION CHECKLIST
# Before submitting each zoom-in:
# 2. Verify 3-second minimum spacing to jump cut
# 3. Verify sentence is NOT in b-roll [ ]
# 4. Verify text exists EXACTLY as quoted
# 5. Verify sentence numbers exist
# 6. Check transition sentence is valid
# 7. Confirm no b-roll between zoom and transition
# 8. Validate total zoom-in count matches floor(minutes)

# # Examples

# Example 1:
# {
#   "zoom_moments": [
#     {
#       "sentence_number": 28,
#       "zoom_in_phrase": "but this CHANGES EVERYTHING",
#       "reason": "Important concept signified by conjunction + emphasized phrase",
#       "transition_sentence_number": 29,
#       "transition_sentence_word": "Let's",
#       "transition_reason": "Start of next sentence, 3second rule is maintained, marks shift in direction"
#     }
#   ]
# }

# Example 2:
# {
#   "zoom_moments": [
#     {
#       "sentence_number": 15,
#       "zoom_in_phrase": "SUCCESS",
#       "reason": "High emphasis word in capital letters followed by 2-second pause",
#       "transition_sentence_number": 17,
#       "transition_sentence_word": "concluded",
#       "transition_reason": "Marks end of idea in sentence 15,3 second rule is maintained, ensures natural flow"
#     }
#   ]
# }

#             """
        
#         self.system_prompt = """You are an intelligent assistant who helps to identify zoom-in moments in a video transcript.

# # ABSOLUTE CRITICAL RULES - MUST BE FOLLOWED WITHOUT EXCEPTION


# 1. PRECISE TIMING AND DISTRIBUTION REQUIREMENTS
# - MANDATORY: EXACTLY ONE zoom-in per minute of video duration
#   * 5-minute video = EXACTLY 5 zoom-ins (unless impossible due to b-rolls)
#   * 10-minute video = EXACTLY 10 zoom-ins (unless impossible due to b-rolls)
# - REQUIRED SPACING:
#   * MINIMUM 3-second interval between zoom-in completion and jump cut
#   * Distribute remaining zoom-ins evenly across available non-b-roll segments
# - VERIFICATION STEPS:
#   1. Calculate total video duration in minutes
#   2. Identify all b-roll segments and mark as unavailable
#   3. Count available spaces for zoom-ins
#   4. If target number cannot be met, document reason and maximize even distribution

# 2. B-ROLL PROTECTION RULE
# - ⚠️ CRITICAL: B-roll segments are STRICTLY PROTECTED ZONES
# - NO zoom-ins or jump cuts may occur within or overlap ANY b-roll segment
# - Before placing ANY zoom-in or jump cut, VERIFY:
#   * The zoom-in start point is outside b-roll
#   * The entire zoom-in duration is outside b-roll
#   * The jump cut point is outside b-roll
#   * The space between zoom-in and jump cut is outside b-roll
# - If ANY part would overlap with b-roll, the zoom-in MUST be relocated or removed


# 3. TRANSITION TIMING PROTOCOL
# - Jump Cut Placement Rules:
#   * MUST occur at complete end of sentence/idea
#   * NEVER place immediately after zoom-in
#   * If zoom-in ends with sentence:
#     - Skip next immediate sentence
#     - Place jump cut at natural break point 2+ sentences later
#     - Ensure 3-second minimum spacing rule is met
#   * VERIFY all spacing requirements before finalizing

# # TRANSCRIPT FORMAT UNDERSTANDING
# - **Capitalized words** indicate emphasis in the audio
# - **Pauses** are denoted by brackets, e.g., `[...s]` indicating a number of seconds
# - **Start and End Times** of each sentence are provided at the end of each sentence
# - **B-roll segments** are marked with squared brackets `[ ]`

# # Zoom-in Key Indicators (Ranked by Descending Priority)
# 1. **Keywords/Phrases**: Look for capitalized words or phrases that signify major importance, especially if they are **followed by a silence** or pause
# 2. **Important Concepts**: Phrases beginning with conjunctions such as "but...", "and...", "so...", or "if..."
# 3. **Emotional Questions**: Questions that convey strong emotion or enthusiasm
# 4. **Exclamations**: Statements like "This is incredible!" that carry exclamation
# 5. **General Emphasized Words**: Capitalized words or phrases not followed by a silence

# # MANDATORY ANALYSIS PROCEDURE

# 1. PRE-ANALYSIS CHECKLIST:
#    - Mark all b-roll segments as exclusion zones
#    - Calculate total required zoom-ins based on video duration
#    - Map available spaces between b-rolls
#    - Verify minimum 3-second spacing availability

# 2. IDENTIFICATION PHASE:
#    - Scan transcript for priority indicators
#    - Mark potential zoom-in points
#    - Cross-reference with b-roll exclusion zones
#    - Document all viable candidates

# 3. SELECTION AND SPACING PHASE:
#    - Apply priority ranking to candidates
#    - Verify spacing requirements
#    - Check distribution across video duration
#    - Ensure no b-roll conflicts

# 4. TRANSITION POINT ANALYSIS:
#    - Identify natural sentence/idea endings
#    - Verify 3-second minimum spacing
#    - Check for b-roll conflicts
#    - Document transition rationale

# # Emphasized Moments to Consider

# - **Zoom-in at Emotional Shifts**: 
#   - When a shift in tone, question, or excitement is indicated by a **capitalized word** or phrase
#   - Examples: **"EXCITING"**, **"IMPORTANT"**, **"SIGNIFICANT"**
  
# - **Zoom-in after Pauses**:
#   - When a **pause or silence** follows a keyword or important phrase
#   - Must verify pause doesn't overlap with b-roll

# - **Use of Conjunctions and Emotional Words**:
#   - Phrases starting with **"BUT"**, **"SO"**, **"AND"**, or **"IF"**
#   - Must be followed by significant content
  
# - **Exclamations**:
#   - Phrases with **exclamation points**
#   - Strong emphasis phrases

# # Output Format

# Return a JSON object with these fields: {"zoom_moments": [{fields}]}
# - **sentence_number**: The sentence number,INDICATED AT THE START OF SENTENCE, where the zoom-in occurs.
# - **zoom_in_phrase**: The specific word/phrase exactly as written in the transcript where the zoom-in starts
# - **reason**: Why this keyword/phrase was chosen (explain based on provided priorities)
# - **transition_sentence_number**: The sentence number, INDICATED AT THE START OF SENTENCE, where the jump cut transition occurs, .
# - **transition_sentence_word**: The exact word/phrase exactly as written in the transcript where the jump cut begins
# - **transition_reason**: Explanation for why this word marks the best transition point

# # Examples

# Example 1:
# {
#   "zoom_moments": [
#     {
#       "sentence_number": 28,
#       "zoom_in_phrase": "but this CHANGES EVERYTHING",
#       "reason": "Important concept signified by conjunction + emphasized phrase",
#       "transition_sentence_number": 29,
#       "transition_sentence_word": "Let's",
#       "transition_reason": "Start of next sentence, 3second rule is maintained, marks shift in direction"
#     }
#   ]
# }

# Example 2:
# {
#   "zoom_moments": [
#     {
#       "sentence_number": 15,
#       "zoom_in_phrase": "SUCCESS",
#       "reason": "High emphasis word in capital letters followed by 2-second pause",
#       "transition_sentence_number": 17,
#       "transition_sentence_word": "concluded",
#       "transition_reason": "Marks end of idea in sentence 15,3 second rule is maintained, ensures natural flow"
#     }
#   ]
# }"""

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
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
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
            
          message = self.client.messages.create(
              model=self.model_name,
              max_tokens=4000,
              temperature=0.7,
              top_p=0.9,
              system=[{
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"}
      }],
              messages=messages,
          )
          out = self.extract_json(message.content[0].text)
          predictions.append(out)

        return predictions
