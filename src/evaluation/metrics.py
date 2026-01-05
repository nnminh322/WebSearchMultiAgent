import numpy as np
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import OpenAI

class AgentMetrics:
    def __init__(self, provider_model: str = "gpt-4o"):
        self.provider = OpenAI(model_engine=provider_model)
        # Trace evaluation provider (dùng model mạnh để chấm điểm plan)
        self.trace_provider = OpenAI(model_engine="gpt-4o") 

    def get_groundedness(self):
        return (
            Feedback(
                self.provider.groundedness_measure_with_cot_reasons, name="Groundedness"
            )
            .on({
                "source": Selector(
                    span_type=SpanAttributes.SpanType.RETRIEVAL,
                    span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                    collect_list=True,
                )
            })
            .on_output()
        )

    def get_answer_relevance(self):
        return (
            Feedback(self.provider.relevance_with_cot_reasons, name="Answer Relevance")
            .on_input()
            .on_output()
        )

    def get_context_relevance(self):
        return (
            Feedback(
                self.provider.context_relevance_with_cot_reasons, name="Context Relevance"
            )
            .on({
                "question": Selector(
                    span_type=SpanAttributes.SpanType.RETRIEVAL,
                    span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
                )
            })
            .on({
                "context": Selector(
                    span_type=SpanAttributes.SpanType.RETRIEVAL,
                    span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                    collect_list=False,
                )
            })
            .aggregate(np.mean) # type: ignore
        )

    def get_trace_metrics(self):
        """Trả về list các metrics đánh giá quy trình (Plan, Logic, Efficiency)."""
        f_plan_quality = Feedback(
            self.trace_provider.plan_quality_with_cot_reasons, name="Plan Quality"
        ).on({"trace": Selector(trace_level=True)})

        f_plan_adherence = Feedback(
            self.trace_provider.plan_adherence_with_cot_reasons, name="Plan Adherence"
        ).on({"trace": Selector(trace_level=True)})

        f_execution_efficiency = Feedback(
            self.trace_provider.execution_efficiency_with_cot_reasons, name="Execution Efficiency"
        ).on({"trace": Selector(trace_level=True)})

        f_logical_consistency = Feedback(
            self.trace_provider.logical_consistency_with_cot_reasons, name="Logical Consistency"
        ).on({"trace": Selector(trace_level=True)})

        return [f_plan_quality, f_plan_adherence, f_execution_efficiency, f_logical_consistency]

    def get_all_feedbacks(self):
        """Tổng hợp tất cả metrics."""
        return [
            self.get_answer_relevance(),
            self.get_context_relevance(),
            self.get_groundedness(),
            *self.get_trace_metrics()
        ]