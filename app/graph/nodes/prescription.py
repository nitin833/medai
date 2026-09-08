"""Prescription node — drafts medicine recommendations from DrugCentral."""

from app.graph.state import MedAIState
from app.models.llm import chat
from app.retrieval.qdrant_client import get_qdrant_client
from app.config import COLLECTION_MEDICAL

DISEASE_ALIASES: dict[str, list[str]] = {
    "pneumonia": ["pneumonia", "pneumonitis", "lung infection"],
    "tuberculosis": ["tuberculosis", "tb"],
    "brain tumor": ["brain tumor", "brain neoplasm", "glioma", "meningioma"],
    "skin disease": ["skin disease", "dermatitis", "eczema", "psoriasis"],
    "diabetic retinopathy": ["diabetic retinopathy", "retinopathy"],
    "bone fracture": ["fracture", "bone fracture"],
    "chest abnormality": ["chest abnormality", "lung disease", "cardiomegaly"],
}


def _search_terms(disease: str) -> list[str]:
    lower = disease.lower()
    terms = {lower}
    for key, aliases in DISEASE_ALIASES.items():
        if key in lower or lower in key:
            terms.update(aliases)
    return list(terms)


def _lookup_drugs_for_disease(disease: str) -> list[dict]:
    client = get_qdrant_client()
    results, _ = client.scroll(
        collection_name=COLLECTION_MEDICAL,
        scroll_filter={"must": [{"key": "type", "match": {"value": "drug"}}]},
        limit=500,
    )

    terms = _search_terms(disease)
    candidates = []
    seen = set()

    for point in results:
        payload = point.payload or {}
        drug_name = payload.get("drug")
        if not drug_name or drug_name in seen:
            continue

        diseases = payload.get("diseases") or []
        diseases_lower = [d.lower() for d in diseases]
        matched = any(
            term in dl or dl in term
            for term in terms
            for dl in diseases_lower
        )
        if matched:
            seen.add(drug_name)
            candidates.append({
                "drug": drug_name,
                "diseases": diseases,
                "source": payload.get("source"),
            })

    return candidates[:10]


def prescription_node(state: MedAIState) -> dict:
    disease = state.get("detected_disease")
    if not disease:
        return {}

    drugs = _lookup_drugs_for_disease(disease)
    drug_names = [d["drug"] for d in drugs if d.get("drug")]
    rag_context = state.get("rag_context") or ""

    if drug_names:
        drug_list = "\n".join(f"- {name}" for name in drug_names[:5])
        llm_draft = chat(
            "You are a clinical pharmacist assistant. Draft a preliminary prescription "
            "outline using ONLY the listed drugs and retrieved treatment context. "
            "Include disclaimer that this requires clinician approval.",
            f"Disease: {disease}\n\nCandidate drugs:\n{drug_list}\n\nTreatment context:\n{rag_context[:2000]}",
            max_tokens=400,
            temperature=0.1,
        )
        draft = llm_draft
    else:
        draft = (
            f"No DrugCentral matches found for {disease}. "
            "Consult Merck/MedlinePlus treatment guidelines from retrieved context."
        )

    return {
        "candidate_drugs": drugs,
        "prescription_draft": draft,
    }
