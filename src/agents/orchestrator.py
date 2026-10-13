"""Wires Proposer -> Contestador -> Juiz into one pipeline call.

This is a plain sequential orchestrator, not a LangGraph graph. That's a
deliberate ordering choice, not a final one: the TCC doc ("Arquitetura
Proposta", Camada de Orquestração) names LangGraph as the provisional
choice for exactly the reason that matters once this becomes multi-round
(a Juiz "unclear" looping back to the Contestador for another pass, with
explicit state and a round cap) — but there is no reason to pull in a
graph framework before the three agents' prompts and parsing are even
working. Get `run_pipeline` below solid and tested first (it already
takes a `max_rounds` parameter so the call sites don't change), then swap
its internals for a LangGraph `StateGraph` with the same three nodes plus
a conditional edge on `Verdict.UNCLEAR`.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.contestador import ContestadorAgent
from src.agents.juiz import JuizAgent
from src.agents.llm_client import LLMClient
from src.agents.proposer import ProposerAgent
from src.schemas import CandidatePair, PipelineResult, Verdict


@dataclass
class AdversarialPipeline:
    proposer: ProposerAgent
    contestador: ContestadorAgent
    juiz: JuizAgent
    max_rounds: int = 1  # >1 is a future extension point; see module docstring

    @classmethod
    def from_single_client(cls, llm: LLMClient, max_rounds: int = 1) -> "AdversarialPipeline":
        """Convenience constructor for the common case: all three agents
        share one LLM client/provider. Pass distinct clients directly to
        the dataclass constructor instead if you want, e.g., a cheaper
        model for the Proposer and a stronger one for the Juiz.
        """
        return cls(
            proposer=ProposerAgent(llm=llm),
            contestador=ContestadorAgent(llm=llm),
            juiz=JuizAgent(llm=llm),
            max_rounds=max_rounds,
        )

    def run(self, pair: CandidatePair, baseline_threshold: float = 0.4190) -> PipelineResult:
        proposer_output = self.proposer.run(pair, threshold=baseline_threshold)
        contestador_output = self.contestador.run(pair, proposer_output)
        juiz_output = self.juiz.run(pair, proposer_output, contestador_output, rounds_used=1)

        # Placeholder for the multi-round extension: if juiz_output.verdict
        # is UNCLEAR and self.max_rounds > 1, this is where you'd send the
        # Juiz's explanation back to the Contestador for a second pass
        # before finalizing. Left as a single round for now so the first
        # version is simple to validate end to end.
        if juiz_output.verdict == Verdict.UNCLEAR and self.max_rounds > 1:
            pass  # TODO(round 2): feed juiz_output.explanation back into contestador.run

        return PipelineResult(
            pair=pair,
            proposer=proposer_output,
            contestador=contestador_output,
            juiz=juiz_output,
        )
