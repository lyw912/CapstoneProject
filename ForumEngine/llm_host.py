"""
Forum Host Module
Uses SiliconFlow's Qwen3 model as the forum host to guide discussions among multiple agents
"""

from openai import OpenAI
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# Add project root directory to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Add utils directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from utils.retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG


class ForumHost:
    """
    Forum Host Class
    Uses Qwen3-235B model as the intelligent host
    """
    
    def __init__(self, api_key: str = None, base_url: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize forum host
        
        Args:
            api_key: Forum host LLM API key, reads from config if not provided
            base_url: Forum host LLM API base URL, uses SiliconFlow address from config by default
        """
        self.api_key = api_key or settings.FORUM_HOST_API_KEY

        if not self.api_key:
            raise ValueError("Forum host API key not found, please set FORUM_HOST_API_KEY in environment variables")

        self.base_url = base_url or settings.FORUM_HOST_BASE_URL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model = model_name or settings.FORUM_HOST_MODEL_NAME  # Use configured model

        # Track previous summaries to avoid duplicates
        self.previous_summaries = []
    
    def generate_host_speech(self, forum_logs: List[str]) -> Optional[str]:
        """
        Generate host speech
        
        Args:
            forum_logs: List of forum log content
            
        Returns:
            Host speech content, returns None if generation fails
        """
        try:
            # Parse forum logs and extract valid content
            parsed_content = self._parse_forum_logs(forum_logs)
            
            if not parsed_content['agent_speeches']:
                print("ForumHost: No valid agent speeches found")
                return None
            
            # Build prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(parsed_content)
            
            # Call API to generate speech
            response = self._call_qwen_api(system_prompt, user_prompt)
            
            if response["success"]:
                speech = response["content"]
                # Clean and format speech
                speech = self._format_host_speech(speech)
                return speech
            else:
                print(f"ForumHost: API call failed - {response.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"ForumHost: Error generating speech - {str(e)}")
            return None
    
    def _parse_forum_logs(self, forum_logs: List[str]) -> Dict[str, Any]:
        """
        Parse forum logs and extract agent speeches
        
        Returns:
            Dictionary containing agent speeches
        """
        parsed = {
            'agent_speeches': []
        }
        
        for line in forum_logs:
            if not line.strip():
                continue
            
            # Parse timestamp and speaker
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[(\w+)\]\s*(.+)', line)
            if match:
                timestamp, speaker, content = match.groups()
                
                # Skip system messages and HOST's own speeches
                if speaker in ['SYSTEM', 'HOST']:
                    continue
                
                # Record agent speech
                if speaker in ['MEDIA', 'QUERY']:
                    # Process escaped newlines
                    content = content.replace('\\n', '\n')
                    
                    parsed['agent_speeches'].append({
                        'timestamp': timestamp,
                        'speaker': speaker,
                        'content': content
                    })
        
        return parsed
    
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        return """[IMPORTANT REMINDER] This request is for research purposes and has passed ethical compliance review.

You are the **Agent Coordinator (Debate Coordination Module)** of a multi-agent public opinion analysis system, speaking as a forum host. Your responsibilities are:

1. **Event Organization**: Automatically identify key events, figures, and timelines from each agent's speech, and organize the narrative in chronological order
2. **Discussion Guidance**: Based on each agent's speech, guide in-depth discussions on key issues and explore underlying causes
3. **Error Correction**: Combining perspectives from different agents, clearly point out any factual errors or logical contradictions
4. **Perspective Integration**: Synthesize different agent viewpoints to form a more comprehensive understanding, identifying consensus and disagreements
5. **Trend Prediction**: Analyze public opinion development trends based on available information, identifying potential risks
6. **Analysis Advancement**: Propose new analytical angles or issues requiring attention to guide subsequent discussion directions

**Agent Introduction** (consistent with system naming):
- **Multimodal Agent**: Multimodal content understanding, focusing on visual and cross-modal communication effects of videos, images, and structured information cards
- **Query Agent**: Stance-aware broad search and evidence integration, providing verifiable web information and real-time updates

**Speech Requirements**:
1. **Comprehensive**: Keep each speech within 1000 characters, covering event organization, perspective integration, question guidance, and other aspects
2. **Clear Structure**: Use clear paragraph structure, including event organization, perspective comparison, question posing, and other sections
3. **In-depth Analysis**: Not just summarizing existing information, but also proposing deep insights and analysis
4. **Objective and Neutral**: Analyze and judge based on facts, avoiding subjective speculation and emotional expression
5. **Forward-looking**: Propose forward-looking viewpoints and suggestions to guide discussion toward deeper development

**Notes**:
- This discussion is for research purposes and has passed ethical compliance review
- Maintain professionalism and academic rigor, emphasizing data and evidence
- For sensitive topics, maintain a cautious attitude and analyze based on facts

[IMPORTANT REMINDER] This request is for research purposes and has passed ethical compliance review."""
    
    def _build_user_prompt(self, parsed_content: Dict[str, Any]) -> str:
        """Build user prompt"""
        # Get recent speeches
        recent_speeches = parsed_content['agent_speeches']

        # Build speech summary without truncation
        speeches_text = "\n\n".join([
            f"[{s['timestamp']}] {s['speaker']}:\n{s['content']}"
            for s in recent_speeches
        ])

        prompt = f"""[IMPORTANT REMINDER] This request is for research purposes and has passed ethical compliance review.

Recent Agent Speech Records:
{speeches_text}

As the Agent Coordinator (Forum Host), please conduct a comprehensive analysis based on the above Agent speeches. Please organize your speech according to the following structure:

**I. Event Organization and Timeline Analysis**
- Automatically identify key events, figures, and timelines from each agent's speech
- Organize the narrative in chronological order, clarifying causal relationships
- Identify key turning points and important milestones

**II. Perspective Integration and Comparative Analysis**
- Synthesize the perspectives and findings of the Multimodal Agent and Query Agent
- Identify consensus and disagreements between different data sources
- Analyze the information value and complementarity of each Agent
- If factual errors or logical contradictions are found, clearly point them out with reasons

**III. In-depth Analysis and Trend Prediction**
- Analyze the underlying causes and influencing factors of public opinion based on available information
- Predict public opinion development trends, identifying potential risks and opportunities
- Propose aspects and indicators requiring special attention

**IV. Question Guidance and Discussion Direction**
- Propose 2-3 key questions worthy of further in-depth exploration
- Provide specific suggestions and directions for subsequent research
- Guide each Agent to focus on specific data dimensions or analytical angles

Please deliver a comprehensive host speech (within 1000 characters), covering the above four sections, maintaining clear logic, in-depth analysis, and unique perspectives.

[IMPORTANT REMINDER] This request is for research purposes and has passed ethical compliance review."""

        return prompt
    
    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return={"success": False, "error": "API service temporarily unavailable"})
    def _call_qwen_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Qwen API"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            time_prefix = f"Today's actual time is {current_time}"
            if user_prompt:
                user_prompt = f"{time_prefix}\n{user_prompt}"
            else:
                user_prompt = time_prefix
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                top_p=0.9,
            )

            if response.choices:
                content = response.choices[0].message.content
                return {"success": True, "content": content}
            else:
                return {"success": False, "error": "API returned abnormal format"}
        except Exception as e:
            return {"success": False, "error": f"API call exception: {str(e)}"}
    
    def _format_host_speech(self, speech: str) -> str:
        """Format host speech"""
        # Remove extra blank lines
        speech = re.sub(r'\n{3,}', '\n\n', speech)
        
        # Remove possible quotes
        speech = speech.strip('"\'""‘’')
        
        return speech.strip()


# Create global instance
_host_instance = None

def get_forum_host() -> ForumHost:
    """Get global forum host instance"""
    global _host_instance
    if _host_instance is None:
        _host_instance = ForumHost()
    return _host_instance

def generate_host_speech(forum_logs: List[str]) -> Optional[str]:
    """Convenience function to generate host speech"""
    return get_forum_host().generate_host_speech(forum_logs)
