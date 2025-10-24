"""
Model Approval Workflow - Simplified Educational Version

Demonstrates automated approval gates for model promotion:
- Performance thresholds (minimum accuracy/AUC)
- Bias constraints (fairness across demographics)
- Business rules (regulatory compliance)
- Manual review process
"""

from datetime import datetime


class ModelApprovalWorkflow:
    """Manages model promotion approval process."""

    def __init__(self, config):
        """
        Initialize approval workflow.

        Args:
            config: Approval configuration (thresholds, rules)
        """
        self.config = config
        self.min_performance = config.get('min_performance_improvement', 0.02)
        self.max_bias_disparity = config.get('max_bias_disparity', 0.10)
        self.require_manual_approval = config.get('require_manual_approval', True)

    def evaluate_promotion(self, challenger_metrics, champion_metrics, bias_results):
        """
        Evaluate if challenger model should be promoted.

        Approval Criteria:
        1. Performance: Challenger improves primary metric by ≥2%
        2. Fairness: All demographic groups have <10% disparity
        3. Stability: No performance degradation on any metric
        4. Manual Review: Human approver signs off (optional)

        Args:
            challenger_metrics: New model performance
            champion_metrics: Current production model performance
            bias_results: Fairness analysis results

        Returns:
            dict: Approval decision with reasoning
        """
        decision = {
            'approved': False,
            'checks': {},
            'reason': '',
            'timestamp': datetime.now().isoformat()
        }

        # Check 1: Performance improvement
        performance_check = self._check_performance(
            challenger_metrics,
            champion_metrics
        )
        decision['checks']['performance'] = performance_check

        # Check 2: Fairness constraints
        fairness_check = self._check_fairness(bias_results)
        decision['checks']['fairness'] = fairness_check

        # Check 3: Stability (no metric degraded)
        stability_check = self._check_stability(
            challenger_metrics,
            champion_metrics
        )
        decision['checks']['stability'] = stability_check

        # Overall decision
        all_automated_checks_pass = all([
            performance_check['passed'],
            fairness_check['passed'],
            stability_check['passed']
        ])

        if not all_automated_checks_pass:
            decision['approved'] = False
            decision['reason'] = "Failed automated checks"
            return decision

        # Manual approval (if required)
        if self.require_manual_approval:
            decision['approved'] = False
            decision['reason'] = "Awaiting manual approval"
            decision['requires_manual_approval'] = True
        else:
            decision['approved'] = True
            decision['reason'] = "All checks passed"

        return decision

    def _check_performance(self, challenger_metrics, champion_metrics):
        """
        Check if challenger improves performance sufficiently.

        Example:
        - Champion AUC: 0.85
        - Challenger AUC: 0.88
        - Improvement: +0.03 (3.5%)
        - Threshold: 0.02 (2%)
        - Result: PASS
        """
        champion_auc = champion_metrics.get('roc_auc', 0)
        challenger_auc = challenger_metrics.get('roc_auc', 0)

        improvement = challenger_auc - champion_auc
        improvement_pct = (improvement / champion_auc) * 100 if champion_auc > 0 else 0

        passed = improvement >= self.min_performance

        return {
            'passed': passed,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'threshold': self.min_performance,
            'message': f"{'✓' if passed else '✗'} Performance improvement: {improvement_pct:.1f}%"
        }

    def _check_fairness(self, bias_results):
        """
        Check if model meets fairness constraints.

        Fairness Metrics:
        - Demographic Parity: Equal positive prediction rates across groups
        - Equal Opportunity: Equal true positive rates across groups
        - Predictive Parity: Equal precision across groups

        Example:
        - Male positive rate: 45%
        - Female positive rate: 48%
        - Disparity: 3% (within 10% threshold)
        - Result: PASS
        """
        if not bias_results:
            return {
                'passed': True,
                'message': "No bias analysis available"
            }

        # Calculate maximum disparity across all groups
        disparities = []
        for group_name, metrics in bias_results.items():
            positive_rate = metrics.get('positive_prediction_rate', 0)
            disparities.append(positive_rate)

        if not disparities:
            return {'passed': True, 'message': "No bias metrics"}

        max_disparity = max(disparities) - min(disparities)
        passed = max_disparity <= self.max_bias_disparity

        return {
            'passed': passed,
            'max_disparity': max_disparity,
            'threshold': self.max_bias_disparity,
            'message': f"{'✓' if passed else '✗'} Bias disparity: {max_disparity:.1%}"
        }

    def _check_stability(self, challenger_metrics, champion_metrics):
        """
        Check that no metrics significantly degraded.

        Even if overall performance improves, check that individual
        metrics (precision, recall) didn't drop too much.

        Example:
        - Precision: 0.82 → 0.85 (improved)
        - Recall: 0.91 → 0.89 (degraded by 2.2%, within 5% tolerance)
        - Result: PASS
        """
        tolerance = 0.05  # Allow 5% degradation in individual metrics

        stability_issues = []

        for metric_name in ['precision', 'recall', 'f1']:
            champion_value = champion_metrics.get(metric_name, 0)
            challenger_value = challenger_metrics.get(metric_name, 0)

            if champion_value > 0:
                degradation = (champion_value - challenger_value) / champion_value

                if degradation > tolerance:
                    stability_issues.append({
                        'metric': metric_name,
                        'degradation_pct': degradation * 100
                    })

        passed = len(stability_issues) == 0

        return {
            'passed': passed,
            'issues': stability_issues,
            'message': f"{'✓' if passed else '✗'} Stability check"
        }

    def create_approval_request(self, model_name, model_version, decision):
        """
        Create approval request for manual review.

        In production, this would:
        - Create ticket in approval system (JIRA, ServiceNow)
        - Notify reviewers via email/Slack
        - Generate approval form with model metrics
        - Track approval status in database

        Args:
            model_name: Name of model
            model_version: Version to promote
            decision: Automated decision results

        Returns:
            dict: Approval request details
        """
        approval_request = {
            'model_name': model_name,
            'model_version': model_version,
            'request_date': datetime.now().isoformat(),
            'automated_checks': decision['checks'],
            'status': 'pending',
            'approvers': ['data_science_lead', 'compliance_officer'],
            'approval_url': f"https://approval-system.com/requests/{model_version}"
        }

        # Log to database
        self._log_approval_request(approval_request)

        return approval_request

    def _log_approval_request(self, request):
        """Log approval request to audit database."""
        # In production: INSERT INTO approval_requests ...
        pass


def main():
    """Demo approval workflow."""
    config = {
        'min_performance_improvement': 0.02,
        'max_bias_disparity': 0.10,
        'require_manual_approval': True
    }

    workflow = ModelApprovalWorkflow(config)

    # Challenger model metrics
    challenger_metrics = {
        'roc_auc': 0.88,
        'precision': 0.85,
        'recall': 0.90,
        'f1': 0.87
    }

    # Current champion metrics
    champion_metrics = {
        'roc_auc': 0.85,
        'precision': 0.83,
        'recall': 0.91,
        'f1': 0.87
    }

    # Bias analysis results
    bias_results = {
        'sex_male': {'positive_prediction_rate': 0.45},
        'sex_female': {'positive_prediction_rate': 0.48}
    }

    # Evaluate promotion
    decision = workflow.evaluate_promotion(
        challenger_metrics,
        champion_metrics,
        bias_results
    )

    print("Approval Decision:")
    print(f"  Approved: {decision['approved']}")
    print(f"  Reason: {decision['reason']}")
    print("\nAutomated Checks:")
    for check_name, check_result in decision['checks'].items():
        print(f"  {check_name}: {check_result['message']}")


if __name__ == '__main__':
    main()
