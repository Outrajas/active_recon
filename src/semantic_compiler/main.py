# src/semantic_compiler/main.py
import sys, os, io, contextlib
from semantic_compiler.utils.config import load_config
from semantic_compiler.utils.vsa import VSAStore
from semantic_compiler.llm.extractor import ExtractorLLM
from semantic_compiler.llm.entity_cleanup import EntityCleanup
from semantic_compiler.reasoning.diagnostic_engine import DiagnosticEngine
from semantic_compiler.pipeline.layout_segmentation import LayoutSegmentationStage
from semantic_compiler.pipeline.document_structure_graph import DocumentStructureGraph
from semantic_compiler.pipeline.extraction import ExtractionStage
from semantic_compiler.pipeline.local_trees import LocalTreesStage
from semantic_compiler.pipeline.hierarchical_memory import HierarchicalMemoryStage
from semantic_compiler.pipeline.global_graph import GlobalGraphStage
from semantic_compiler.pipeline.consolidation import ConsolidationStage
from semantic_compiler.pipeline.reasoning import ReasoningStage
from semantic_compiler.pipeline.output import OutputStage
from semantic_compiler.pipeline.reconstruction import ReconstructionStage
from semantic_compiler.pipeline.concept_consolidation import ConceptConsolidationStage
from semantic_compiler.pipeline.coverage_auditor import CoverageAuditor

class SemanticCompiler:
    def __init__(self, config_path="config/settings.yaml"):
        self.config = load_config(config_path)
        self.vsa = VSAStore()
        self.llm = ExtractorLLM(device="cpu")
        self.engine = DiagnosticEngine(self.config['reasoning']['probability_threshold'])
        self.output_mode = self.config['output']['mode']

    def run(self, input_path: str, reconstruct: bool = False, concepts: bool = False, audit: bool = False):
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.pdf':
            layout_stage = LayoutSegmentationStage()
            blocks = layout_stage.run(input_path)
            semantic_blocks = [b for b in blocks if b['document_role'] in ('PARAGRAPH', 'ABSTRACT')]
            metadata_blocks = [b for b in blocks if b['document_role'] not in ('PARAGRAPH', 'ABSTRACT')]
            chunks = {}
            chunk_attrs = {}
            for i, b in enumerate(semantic_blocks):
                chunks[i] = b['text']
                chunk_attrs[i] = {'page': b['page'], 'y': b['y']}
            extra_chunks = {}
            for i, b in enumerate(metadata_blocks):
                cid = f"meta_{i}"
                extra_chunks[cid] = b
                chunk_attrs[cid] = {'page': b['page'], 'y': b['y']}
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if not paragraphs:
                paragraphs = [text]
            chunks = {i: p for i, p in enumerate(paragraphs)}
            chunk_attrs = {i: {'page': 1, 'y': float(i)} for i in chunks}
            extra_chunks = {}

        extractor = ExtractionStage(self.llm, self.vsa)
        extraction_results = extractor.run(chunks)

        for cid, block in extra_chunks.items():
            extraction_results[cid] = [{
                "text": block['text'],
                "semantic_role": "NONE",
                "document_role": block['document_role'],
                "confidence": 1.0,
                "source_chunk": cid,
                "source_offsets": [0, len(block['text'])],
                "triples": [],
                "entities": []
            }]

        cleaner = EntityCleanup()
        extraction_results = cleaner.clean(extraction_results)

        all_chunk_texts = {}
        all_chunk_texts.update({i: chunks[i] for i in chunks})
        for cid, block in extra_chunks.items():
            all_chunk_texts[cid] = block['text']

        local = LocalTreesStage()
        local_graphs = local.run(extraction_results, chunk_texts=all_chunk_texts, chunk_attrs=chunk_attrs)

        memory = HierarchicalMemoryStage()
        merged = memory.run(local_graphs)

        # Suppress GlobalGraphStage output in audit mode to keep stdout clean
        if audit:
            with open(os.devnull, 'w') as fnull:
                with contextlib.redirect_stdout(fnull):
                    GlobalGraphStage().run(merged)
        else:
            GlobalGraphStage().run(merged)

        cons = ConsolidationStage(threshold=0.75, device="cpu")
        consolidated = cons.run(merged)

        if concepts:
            concept_stage = ConceptConsolidationStage(threshold=0.8, device="cpu")
            consolidated = concept_stage.run(consolidated)

        reason = ReasoningStage(self.engine)
        reasoned = reason.run(consolidated)

        if audit:
            auditor = CoverageAuditor()
            stats = auditor.audit(reasoned)
            import json
            return json.dumps(stats, indent=2)

        if reconstruct:
            recon = ReconstructionStage()
            return recon.run(reasoned)
        else:
            output_stage = OutputStage()
            if self.output_mode == "pseudo-language":
                return output_stage.run_pseudo_language(reasoned)
            else:
                return output_stage.run_latent_injection(reasoned)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m semantic_compiler.main <input.pdf|input.txt> [--reconstruct] [--concepts] [--audit]")
        sys.exit(1)
    reconstruct = "--reconstruct" in sys.argv
    concepts = "--concepts" in sys.argv
    audit = "--audit" in sys.argv
    compiler = SemanticCompiler()
    output = compiler.run(sys.argv[1], reconstruct=reconstruct, concepts=concepts, audit=audit)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print(output)
