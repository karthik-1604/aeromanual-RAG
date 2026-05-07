import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from src.embeddings.vector_store import load_index
from src.pipeline.rag_chain import build_rag_chain

# 20 benchmark questions with ground truth answers
TEST_QUERIES = [
    {
        "question": "What are the inspection requirements for turbine engine hot section components?",
        "ground_truth": "Hot section inspections include visual examination of turbine blades, combustion liners, and transition ducts at manufacturer-specified intervals."
    },
    {
        "question": "Describe the procedure for magneto timing on a reciprocating engine.",
        "ground_truth": "Magneto timing involves setting the piston to TDC, aligning the magneto to fire at specified degrees BTDC, and verifying with a timing light."
    },
    {
        "question": "What safety precautions apply when working on aircraft fuel systems?",
        "ground_truth": "Fuel system work requires grounding the aircraft, eliminating open flames, ensuring proper ventilation, and using approved solvents and PPE."
    },
    {
        "question": "What are the hydraulic system maintenance checks for landing gear?",
        "ground_truth": "Hydraulic checks include inspecting fluid levels, checking for leaks at fittings and seals, testing actuator operation, and verifying pressure settings."
    },
    {
        "question": "How is a compression test performed on a reciprocating engine?",
        "ground_truth": "A differential compression test uses regulated air pressure to check each cylinder, comparing input to output pressure to identify leaking valves or rings."
    },
    {
        "question": "What are the requirements for aircraft battery maintenance?",
        "ground_truth": "Battery maintenance includes checking electrolyte levels, testing capacity, cleaning terminals, checking for corrosion, and ensuring proper charge levels."
    },
    {
        "question": "Explain the process of bleeding aircraft brakes.",
        "ground_truth": "Brake bleeding removes air from hydraulic lines by opening bleeder valves while applying pressure until fluid flows without bubbles."
    },
    {
        "question": "What NDT methods are used for aircraft structural inspection?",
        "ground_truth": "NDT methods include visual inspection, dye penetrant, magnetic particle, eddy current, ultrasonic testing, and radiographic inspection."
    },
    {
        "question": "What are the torque specifications and procedures for cylinder hold-down nuts?",
        "ground_truth": "Cylinder hold-down nuts are torqued in a cross pattern to manufacturer specifications, typically in multiple passes to ensure even seating."
    },
    {
        "question": "How is aircraft corrosion treated and prevented?",
        "ground_truth": "Corrosion treatment involves removing affected material, applying chemical conversion coatings, and protecting surfaces with approved primers and topcoats."
    },
    {
        "question": "What are the inspection criteria for aircraft control cables?",
        "ground_truth": "Control cables are inspected for broken wires, corrosion, kinks, and proper tension. Cables with more than the allowed broken wires must be replaced."
    },
    {
        "question": "Describe the procedure for rigging aircraft flight controls.",
        "ground_truth": "Control rigging involves setting control surfaces to neutral, adjusting cable tensions to specifications, and verifying full travel and stops."
    },
    {
        "question": "What are the requirements for aircraft welding repairs?",
        "ground_truth": "Welding repairs require approved procedures, certified welders, proper preheat treatment, and post-weld inspection including NDT methods."
    },
    {
        "question": "How is an aircraft pitot-static system tested?",
        "ground_truth": "Pitot-static systems are tested using calibrated test equipment to verify altimeter, airspeed, and VSI accuracy within FAA-required tolerances."
    },
    {
        "question": "What are the fire extinguisher requirements for aircraft?",
        "ground_truth": "Aircraft fire extinguishers must be of approved type, properly charged, securely mounted, and inspected at required intervals per regulations."
    },
    {
        "question": "Explain the procedure for checking aircraft tire condition.",
        "ground_truth": "Tire inspection includes checking tread depth, looking for cuts, bulges, or flat spots, verifying proper inflation, and checking for weather cracking."
    },
    {
        "question": "What are the maintenance requirements for aircraft oxygen systems?",
        "ground_truth": "Oxygen system maintenance includes checking cylinder pressure, inspecting masks and regulators, testing flow rates, and following strict no-oil contamination rules."
    },
    {
        "question": "How are aircraft fuel tanks inspected for contamination?",
        "ground_truth": "Fuel tank inspection involves sump draining to check for water and sediment, visual inspection through access panels, and sampling for microbiological contamination."
    },
    {
        "question": "What are the procedures for aircraft propeller inspection?",
        "ground_truth": "Propeller inspection includes checking for nicks, cracks, erosion, blade tracking, hub condition, and verifying torque values on all bolts."
    },
    {
        "question": "Describe the airframe inspection requirements for a 100-hour inspection.",
        "ground_truth": "A 100-hour inspection covers all aircraft systems including engine, airframe, flight controls, landing gear, avionics, and required documentation review."
    },
]


def run_evaluation(output_csv: str = "evaluation_results.csv"):
    print("Loading pipeline...")
    vs = load_index()
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    chain = build_rag_chain(retriever)

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"Running {len(TEST_QUERIES)} test queries...")
    for i, item in enumerate(TEST_QUERIES):
        print(f"  Query {i+1}/{len(TEST_QUERIES)}: {item['question'][:60]}...")
        result = chain.invoke({"query": item["question"]})
        data["question"].append(item["question"])
        data["answer"].append(result["result"])
        data["contexts"].append([d.page_content for d in result["source_documents"]])
        data["ground_truth"].append(item["ground_truth"])

    print("\nRunning RAGAS evaluation...")
    dataset = Dataset.from_dict(data)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

    print("\n=== EVALUATION RESULTS ===")
    print(scores)

    df = scores.to_pandas()
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")
    return scores


if __name__ == "__main__":
    run_evaluation()
