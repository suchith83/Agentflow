"""
Example script demonstrating how to run comprehensive evaluation on the react_sync agent.

This script shows how to:
1. Load evaluation cases
2. Run the agent with evaluation tracking
3. Collect metrics and generate reports
4. Analyze results
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

from agentflow.state import AgentState, Message
from agentflow.evaluation import EvalCase
from agentflow.evaluation.collectors.trajectory_collector import TrajectoryCollector
from agentflow.evaluation.eval_result import EvalReport, EvalCaseResult, CriterionResult

# Import our evaluation configuration
from evaluation_config import (
    create_comprehensive_eval_config,
    get_all_evaluation_cases,
    filter_cases_by_difficulty,
    filter_cases_by_category
)

# Import the agent from react_sync
import sys
react_dir = Path(__file__).parent
sys.path.insert(0, str(react_dir))


class ReactSyncEvaluator:
    """Evaluator for the react_sync agent."""
    
    def __init__(self, app, verbose: bool = True):
        """
        Initialize evaluator.
        
        Args:
            app: Compiled agent graph
            verbose: Whether to print progress
        """
        self.app = app
        self.verbose = verbose
    
    def run_single_case(
        self,
        case: EvalCase,
        collector: TrajectoryCollector | None = None
    ) -> Dict[str, Any]:
        """
        Run a single evaluation case.
        
        Args:
            case: Evaluation case to run
            collector: Optional trajectory collector
        
        Returns:
            Dict containing results and metrics
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Running: {case.name}")
            print(f"ID: {case.id}")
            print(f"{'='*70}\n")
        
        # Prepare input
        inp = {
            "messages": [
                Message.text_message(msg.content, role=msg.role)
                for msg in case.input_messages
            ]
        }
        
        config = {
            "thread_id": f"eval_{case.id}",
            "recursion_limit": 10
        }
        
        # Initialize collector if not provided
        if collector is None:
            collector = TrajectoryCollector()
        
        # Run the agent
        start_time = datetime.now()
        try:
            result = self.app.invoke(inp, config=config)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Collect messages
        if result and "messages" in result:
            for msg in result["messages"]:
                collector.add_message(msg)
        
        # Analyze results
        tool_calls_made = []
        if result and "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, 'tools_calls') and msg.tools_calls:
                    for tc in msg.tools_calls:
                        tool_name = tc.get("function", {}).get("name", "unknown")
                        tool_calls_made.append(tool_name)
        
        # Check tool call accuracy
        expected_tools = set(case.expected_tool_calls) if case.expected_tool_calls else set()
        actual_tools = set(tool_calls_made)
        
        tool_accuracy = 1.0 if expected_tools == actual_tools else (
            len(expected_tools & actual_tools) / max(len(expected_tools | actual_tools), 1)
        )
        
        # Prepare result
        case_result = {
            "case_id": case.id,
            "case_name": case.name,
            "success": success,
            "error": error,
            "execution_time_seconds": execution_time,
            "expected_tool_calls": list(expected_tools),
            "actual_tool_calls": tool_calls_made,
            "tool_call_accuracy": tool_accuracy,
            "message_count": len(result.get("messages", [])) if result else 0,
            "metadata": case.metadata
        }
        
        if self.verbose:
            print(f"✓ Status: {'SUCCESS' if success else 'FAILED'}")
            print(f"✓ Execution time: {execution_time:.3f}s")
            print(f"✓ Tool accuracy: {tool_accuracy:.1%}")
            print(f"✓ Messages generated: {case_result['message_count']}")
            if error:
                print(f"✗ Error: {error}")
        
        return case_result
    
    def run_evaluation_suite(
        self,
        cases: List[EvalCase]
    ) -> Dict[str, Any]:
        """
        Run a complete evaluation suite.
        
        Args:
            cases: List of evaluation cases to run
        
        Returns:
            Dict containing aggregated results
        """
        results = []
        
        print(f"\n{'='*70}")
        print(f"STARTING EVALUATION SUITE")
        print(f"Total cases: {len(cases)}")
        print(f"{'='*70}")
        
        for i, case in enumerate(cases, 1):
            if self.verbose:
                print(f"\nProgress: {i}/{len(cases)}")
            
            result = self.run_single_case(case)
            results.append(result)
        
        # Aggregate results
        total_cases = len(results)
        successful_cases = sum(1 for r in results if r["success"])
        failed_cases = total_cases - successful_cases
        
        total_time = sum(r["execution_time_seconds"] for r in results)
        avg_time = total_time / total_cases if total_cases > 0 else 0
        
        avg_tool_accuracy = sum(r["tool_call_accuracy"] for r in results) / total_cases if total_cases > 0 else 0
        
        # Group by category
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for result in results:
            category = result["metadata"].get("category", "unknown")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        category_stats = {}
        for category, cat_results in by_category.items():
            category_stats[category] = {
                "total": len(cat_results),
                "successful": sum(1 for r in cat_results if r["success"]),
                "avg_tool_accuracy": sum(r["tool_call_accuracy"] for r in cat_results) / len(cat_results)
            }
        
        summary = {
            "total_cases": total_cases,
            "successful_cases": successful_cases,
            "failed_cases": failed_cases,
            "success_rate": successful_cases / total_cases if total_cases > 0 else 0,
            "total_execution_time_seconds": total_time,
            "average_execution_time_seconds": avg_time,
            "average_tool_accuracy": avg_tool_accuracy,
            "by_category": category_stats,
            "detailed_results": results
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"Total Cases:       {summary['total_cases']}")
        print(f"Successful:        {summary['successful_cases']} ({summary['success_rate']:.1%})")
        print(f"Failed:            {summary['failed_cases']}")
        print(f"Avg Execution:     {summary['average_execution_time_seconds']:.3f}s")
        print(f"Total Time:        {summary['total_execution_time_seconds']:.3f}s")
        print(f"Avg Tool Accuracy: {summary['average_tool_accuracy']:.1%}")
        
        print(f"\n{'='*70}")
        print("BY CATEGORY")
        print(f"{'='*70}\n")
        
        for category, stats in summary['by_category'].items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{category.upper()}:")
            print(f"  Total:        {stats['total']}")
            print(f"  Successful:   {stats['successful']} ({success_rate:.1%})")
            print(f"  Tool Accuracy: {stats['avg_tool_accuracy']:.1%}")
            print()
        
        print(f"{'='*70}\n")
    
    def save_results(self, summary: Dict[str, Any], output_file: str):
        """Save results to JSON file."""
        output_path = Path(output_file)
        
        # Add timestamp
        summary["timestamp"] = datetime.now().isoformat()
        summary["evaluation_framework"] = "agentflow"
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Results saved to: {output_path}")


def main():
    """Main evaluation entry point."""
    print("\n" + "="*70)
    print("REACT SYNC AGENT - COMPREHENSIVE EVALUATION")
    print("="*70 + "\n")
    
    # Import the app
    try:
        from react_sync import app
    except ImportError as e:
        print(f"Error importing react_sync: {e}")
        print("Make sure you're running from the react directory")
        return 1
    
    # Create evaluator
    evaluator = ReactSyncEvaluator(app, verbose=True)
    
    # Load evaluation cases
    all_cases = get_all_evaluation_cases()
    
    print(f"Loaded {len(all_cases)} evaluation cases")
    
    # Option 1: Run all cases
    print("\n>>> Running all evaluation cases...")
    summary = evaluator.run_evaluation_suite(all_cases)
    
    # Print summary
    evaluator.print_summary(summary)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    evaluator.save_results(summary, output_file)
    
    # Option 2: Run by category (commented out)
    # weather_cases = filter_cases_by_category(all_cases, "weather")
    # weather_summary = evaluator.run_evaluation_suite(weather_cases)
    
    # Option 3: Run by difficulty (commented out)
    # easy_cases = filter_cases_by_difficulty(all_cases, "easy")
    # easy_summary = evaluator.run_evaluation_suite(easy_cases)
    
    return 0


def run_quick_evaluation():
    """Run a quick evaluation with just a few cases."""
    from react_sync import app
    
    # Just run the first 3 cases
    all_cases = get_all_evaluation_cases()
    quick_cases = all_cases[:3]
    
    evaluator = ReactSyncEvaluator(app, verbose=True)
    summary = evaluator.run_evaluation_suite(quick_cases)
    evaluator.print_summary(summary)
    
    return summary


def run_category_evaluation(category: str):
    """Run evaluation for a specific category."""
    from react_sync import app
    
    all_cases = get_all_evaluation_cases()
    category_cases = filter_cases_by_category(all_cases, category)
    
    if not category_cases:
        print(f"No cases found for category: {category}")
        return None
    
    evaluator = ReactSyncEvaluator(app, verbose=True)
    summary = evaluator.run_evaluation_suite(category_cases)
    evaluator.print_summary(summary)
    
    return summary


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            run_quick_evaluation()
        elif command == "category" and len(sys.argv) > 2:
            category = sys.argv[2]
            run_category_evaluation(category)
        else:
            print("Usage:")
            print("  python run_evaluation.py           # Run full evaluation")
            print("  python run_evaluation.py quick     # Run quick evaluation")
            print("  python run_evaluation.py category weather  # Run specific category")
    else:
        sys.exit(main())
