import os
import json
import base64
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage

load_dotenv("N:/proj/intern-schi/pid.env")

if "GOOGLE_GENAI_USE_VERTEXAI" in os.environ:
    del os.environ["GOOGLE_GENAI_USE_VERTEXAI"]
if "GOOGLE_CLOUD_PROJECT" in os.environ:
    del os.environ["GOOGLE_CLOUD_PROJECT"]

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    API_KEY = os.getenv("KEY")
if not API_KEY:
    raise ValueError("API Key not found! Check your pid.env file.")

print(f" Loaded API Key: {API_KEY[:5]}...{API_KEY[-5:]}")

try:
    print(" Initializing Gemini 3.0 Flash Preview...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=API_KEY,
        temperature=0.2,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    print(" Model Active: gemini-3-flash-preview (Safety Filters: OFF)")
except Exception as e:
    raise ValueError(f"LLM Init Failed: {e}")


# --- 1. DATA MODELS ---
class BoundingBox(BaseModel):
    ymin: int = Field(description="Y min coordinate (0-1000)")
    xmin: int = Field(description="X min coordinate (0-1000)")
    ymax: int = Field(description="Y max coordinate (0-1000)")
    xmax: int = Field(description="X max coordinate (0-1000)")

class Symbol(BaseModel):
    tag: str = Field(description="Tag number (e.g. V-101, TIC-202). If none, use 'Unknown'.")
    type: str = Field(description="Type of symbol (Valve, Instrument, Pump, Vessel, HeatExchanger, Line, etc.)")
    name: str = Field(description="Specific name (e.g. Globe Valve, Control Valve)")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    box_2d: BoundingBox = Field(description="Bounding box in 0-1000 normalized coordinates")

class ExtractionResult(BaseModel):
    symbols: List[Symbol]

class AgentState(TypedDict):
    image_path: str
    detections: List[dict]
    loop_count: int
    detector_threshold: float
    verifier_threshold: float

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def draw_boxes_from_json(image_path: str, detections: List[dict], output_prefix: str):
    """Draws bounding boxes from the detection list."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(" Error loading image for drawing.")
            return

        height, width, _ = img.shape
        count = 0
        
        for d in detections:
            try:
                bbox = d.get('box_2d', {})
                if not bbox: continue
                
                x1 = int(bbox.get('xmin', 0) * width / 1000)
                y1 = int(bbox.get('ymin', 0) * height / 1000)
                x2 = int(bbox.get('xmax', 0) * width / 1000)
                y2 = int(bbox.get('ymax', 0) * height / 1000)
                
                tag = d.get('tag', 'Unknown')
                stype = d.get('type', 'Item')
                conf = d.get('confidence', 0.0)
                
                label = f"{tag} | {stype} ({conf:.2f})"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img, (x1, y1 - 20), (x1 + text_size[0], y1), (0, 255, 0), -1)
                
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                count += 1
            except Exception as e:
                print(f"Skipping a box: {e}")
        
        out_path = f"{output_prefix}_annotated.png"
        cv2.imwrite(out_path, img)
        print(f" Saved annotated image to: {out_path} (Boxes: {count})")
        
    except Exception as e:
        print(f" Drawing Error: {e}")

def agent_detector(state: AgentState):
    print(f"\n [Detector] Loop {state['loop_count'] + 1} | Threshold: {state['detector_threshold']}")
    
    image_path = state["image_path"]
    threshold = state["detector_threshold"]
    existing_detections = state.get("detections", [])
    
    base64_image = encode_image(image_path)

    if existing_detections:
        text_prompt = f"""
        Analyze this P&ID image again to find missed items.
        
        Current Detections: {len(existing_detections)} items found.
        
        Your Mission:
        1. Find missed symbols (small valves, instruments, line tags).
        2. Use SAHI logic (Scanning for small objects).
        3. Keep existing valid symbols.
        4. Return a COMPLETE list of ALL valid symbols found.
        
        Confidence Threshold: > {threshold}.
        """
    else:
        text_prompt = f"""
        Identify ALL P&ID symbols in this image.
        
        Return a JSON list of symbols:
        - tag (e.g. V-101)
        - type (Valve, Pump, Instrument, Line, etc.)
        - name (Globe Valve, etc.)
        - confidence (0.0-1.0)
        - box_2d (0-1000 normalized [ymin, xmin, ymax, xmax])
        
        Be thorough. Cover everything.
        Confidence Threshold: > {threshold}.
        """
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"},
        ]
    )
    
    try:

        structured_llm = llm.with_structured_output(ExtractionResult)
        result = structured_llm.invoke([message])
        
        if not result or not result.symbols:
            print("   -> No symbols found.")
            return {"detections": existing_detections} 

        new_detections = [s.model_dump() for s in result.symbols]
        print(f"   -> Found {len(new_detections)} symbols.")
        return {"detections": new_detections}
        
    except Exception as e:
        print(f" Detector Error: {e}")
        return {"detections": state.get("detections", [])}

def agent_verifier(state: AgentState):
    print(f"✓ [Verifier] Loop {state['loop_count'] + 1} | Threshold: {state['verifier_threshold']}")
    
    image_path = state["image_path"]
    detections = state["detections"]
    threshold = state["verifier_threshold"]
    
    if not detections:
        print("   -> Skipping verification (No detections).")
        return {"detections": []}
    
    base64_image = encode_image(image_path)

    text_prompt = f"""
    Review these {len(detections)} detected symbols.
    
    Verify:
    1. Tags are correct.
    2. Types match the visual symbol.
    3. Bounding boxes are tight and accurate.
    4. Remove false positives below confidence {threshold}.
    
    Return the Corrected List.
    """
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"},
        ]
    )
    
    try:
        structured_llm = llm.with_structured_output(ExtractionResult)
        result = structured_llm.invoke([message])
        
        if not result or not result.symbols:
             print("   -> Verifier returned empty. Keeping originals.")
             return {"detections": detections}
             
        verified = [s.model_dump() for s in result.symbols]
        print(f"   -> Verified count: {len(verified)}")
        return {"detections": verified}
            
    except Exception as e:
        print(f" Verifier Error: {e}")
        return {"detections": detections}

def update_loop_state(state: AgentState):
    current_loop = state["loop_count"] + 1
    if current_loop == 1:
        new_det_thresh = 0.05
        new_ver_thresh = 0.01
    else:
        new_det_thresh = state["detector_threshold"]
        new_ver_thresh = state["verifier_threshold"]

    return {
        "loop_count": current_loop,
        "detector_threshold": new_det_thresh,
        "verifier_threshold": new_ver_thresh
    }

def should_continue(state: AgentState):
    if state["loop_count"] < 2:
        return "continue"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("detector", agent_detector)
workflow.add_node("verifier", agent_verifier)
workflow.add_node("update_state", update_loop_state)
workflow.add_edge(START, "detector")
workflow.add_edge("detector", "verifier")
workflow.add_edge("verifier", "update_state")
workflow.add_conditional_edges("update_state", should_continue, {"continue": "detector", "end": END})
app_graph = workflow.compile()

if __name__ == "__main__":
    image_input = input("\n Drag & drop P&ID image here (or paste path): ").strip().strip('"')
    
    if not os.path.exists(image_input):
        print(" File not found.")
        exit()
        
    print(f"\n Starting Agentic Extraction on: {image_input}")
    
    final_state = app_graph.invoke({
        "image_path": image_input,
        "detections": [],
        "loop_count": 0,
        "detector_threshold": 0.1,
        "verifier_threshold": 0.05
    })
    
    detections = final_state.get("detections", [])
    count = len(detections)
    
    if count > 0:
        base_name = os.path.splitext(os.path.basename(image_input))[0]
        output_dir = os.path.dirname(image_input)
        prefix = os.path.join(output_dir, base_name)
        
        json_path = f"{prefix}_results.json"
        with open(json_path, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f"\n Saved JSON: {json_path}")
        
        df = pd.DataFrame([
            {
                'tag': d.get('tag'),
                'type': d.get('type'),
                'name': d.get('name'),
                'confidence': d.get('confidence'),
                'ymin': d.get('box_2d', {}).get('ymin'),
                'xmin': d.get('box_2d', {}).get('xmin'),
                'ymax': d.get('box_2d', {}).get('ymax'),
                'xmax': d.get('box_2d', {}).get('xmax')
            }
            for d in detections
        ])
        xlsx_path = f"{prefix}_results.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f" Saved Excel: {xlsx_path}")
        
        draw_boxes_from_json(image_input, detections, prefix)
        
    else:
        print("\n No symbols found. Try checking the image resolution or lighting.")

