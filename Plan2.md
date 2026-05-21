1. Five LLM criteria are the same class written five times
HallucinationCriterion, FactualAccuracyCriterion, SafetyCriterion, RubricBasedCriterion, LLMJudgeCriterion -- all do:

build prompt from template
call _run_samples()
average scores
return CriterionResult.success() or .failure()
The only real difference is the prompt template and which JSON fields they read back. A single TemplatedLLMCriterion base class with a prompt_template class attribute and a _parse_response(dict) -> float hook would collapse ~700 lines to ~150.

2. Criterion name mapping is split across two files
evaluator.py has _create_criterion() mapping internal names → classes. eval_config.py has CriteriaConfig.to_dict() / from_dict() mapping user-facing names → internal names. Two separate maps that must stay in sync, and there are aliases ("trajectory_match" and "tool_trajectory_avg_score" both work). A single registry dict in criteria/__init__.py -- {"trajectory": TrajectoryMatchCriterion, ...} -- eliminates both maps entirely. Config and evaluator both import from it.

3. _evaluate_case() is 220 lines and does everything
evaluator.py:456-676 handles graph invocation, multi-turn loop, state accumulation across turns, message deduplication, timing, ExecutionResult construction, and criterion evaluation -- all in one method. Message deduplication logic even appears twice (single-turn path vs multi-turn path). This needs splitting into 3 focused methods: run one turn, accumulate across turns, build the final ExecutionResult.

4. CriteriaConfig is a Pydantic model pretending to be a dict
eval_config.py has items(), keys(), values(), get(), __getitem__, __contains__, __iter__, __len__ bolted onto a Pydantic model for "backward compatibility". That's a dict. The underlying issue is that AgentEvaluator iterates config.criteria like a dict. If you fix the registry (point 2), the evaluator stops needing a dict interface entirely and all those shim methods disappear.

5. BaseCriterion has no shared helpers, so subclasses repeat themselves
Every criterion repeats the same _extract_expected_response() / _extract_question() logic inline. LLMJudgeCriterion defines _extract_question and _extract_expected_response -- but other criteria re-implement the same thing. These belong on BaseCriterion once.

Same for result building -- every criterion manually does:


CriterionResult.success(criterion=self.name, score=..., threshold=self.threshold, ...)
A self._result(score, details, token_usage) helper on the base cuts the noise.

6. TrajectoryCollector is 699 lines doing three unrelated jobs
trajectory_collector.py combines: a PublisherCallback adaptor that converts graph events into messages, the actual TrajectoryCollector that accumulates state, and an EventCollector debug store. These should be three separate classes. The big one is PublisherCallback -- it has complex message extraction logic that's hard to follow because it's buried inside the collector.

Summary table
Problem	Approx lines wasted	Fix
5 near-identical LLM criteria	~700	TemplatedLLMCriterion base with template + parse hook
Dual criterion name maps	~200	Single registry dict
_evaluate_case() god method	~220	3 focused private methods
Dict shims on CriteriaConfig	~80	Remove once registry is in place
Repeated _extract_* in criteria	~60	Move to BaseCriterion
TrajectoryCollector doing 3 jobs	~200	Split into 3 classes
Total: roughly 1,400 lines that are complexity with no functionality. Want me to start on any of these?