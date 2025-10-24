"""
Audit Logging for Community Hospital MLOps Platform

This module provides comprehensive audit trail functionality for tracking
all significant events in the ML lifecycle, including model promotions,
approvals, rejections, drift alerts, and data access.

Compliance: Supports HIPAA, SOC 2, and other regulatory requirements
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import json
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Centralized audit logging for ML operations.
    """

    def __init__(self, db_path: str = "src/governance/audit_log.db"):
        """
        Initialize audit logger.

        Args:
            db_path: Path to SQLite database for audit logs
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_db()

        logger.info(f"Initialized audit logger: {self.db_path}")

    def _initialize_db(self):
        """Initialize SQLite database with audit tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Model lifecycle events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_lifecycle_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                model_name TEXT,
                model_version TEXT,
                run_id TEXT,
                user TEXT,
                details TEXT,
                timestamp TEXT,
                event_hash TEXT
            )
        ''')

        # Data access events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_access_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                dataset_name TEXT,
                user TEXT,
                access_method TEXT,
                records_accessed INTEGER,
                purpose TEXT,
                timestamp TEXT,
                event_hash TEXT
            )
        ''')

        # Drift detection events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                drift_type TEXT,
                drift_detected INTEGER,
                drift_score REAL,
                threshold REAL,
                action_taken TEXT,
                timestamp TEXT,
                event_hash TEXT
            )
        ''')

        # Model approval events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS approval_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                challenger_run_id TEXT,
                champion_run_id TEXT,
                approved INTEGER,
                approver TEXT,
                reason TEXT,
                checks_passed TEXT,
                timestamp TEXT,
                event_hash TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def _calculate_event_hash(self, event_data: Dict) -> str:
        """
        Calculate hash of event for integrity verification.

        Args:
            event_data: Event data dictionary

        Returns:
            str: SHA256 hash
        """
        event_str = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()[:16]

    def log_model_training(
        self,
        model_name: str,
        run_id: str,
        user: str = "system",
        details: Dict = None
    ):
        """
        Log model training event.

        Args:
            model_name: Name of the model
            run_id: MLflow run ID
            user: User who initiated training
            details: Additional details
        """
        event_data = {
            'event_type': 'model_training',
            'model_name': model_name,
            'run_id': run_id,
            'user': user,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }

        event_hash = self._calculate_event_hash(event_data)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO model_lifecycle_events
            (event_type, model_name, run_id, user, details, timestamp, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            'model_training',
            model_name,
            run_id,
            user,
            json.dumps(details),
            event_data['timestamp'],
            event_hash
        ))

        conn.commit()
        conn.close()

        logger.info(f"✓ Logged model training: {model_name} (run: {run_id})")

    def log_model_promotion(
        self,
        model_name: str,
        model_version: str,
        from_alias: str,
        to_alias: str,
        user: str = "system",
        reason: str = None
    ):
        """
        Log model promotion event.

        Args:
            model_name: Name of the model
            model_version: Model version number
            from_alias: Source alias (e.g., 'challenger')
            to_alias: Target alias (e.g., 'champion')
            user: User who approved promotion
            reason: Reason for promotion
        """
        event_data = {
            'event_type': 'model_promotion',
            'model_name': model_name,
            'model_version': model_version,
            'from_alias': from_alias,
            'to_alias': to_alias,
            'user': user,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

        event_hash = self._calculate_event_hash(event_data)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO model_lifecycle_events
            (event_type, model_name, model_version, user, details, timestamp, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            'model_promotion',
            model_name,
            model_version,
            user,
            json.dumps({
                'from_alias': from_alias,
                'to_alias': to_alias,
                'reason': reason
            }),
            event_data['timestamp'],
            event_hash
        ))

        conn.commit()
        conn.close()

        logger.info(f"✓ Logged model promotion: {model_name} v{model_version} ({from_alias} → {to_alias})")

    def log_model_approval(
        self,
        model_name: str,
        challenger_run_id: str,
        champion_run_id: str,
        approved: bool,
        approver: str,
        reason: str,
        checks_passed: Dict
    ):
        """
        Log model approval decision.

        Args:
            model_name: Name of the model
            challenger_run_id: Challenger run ID
            champion_run_id: Champion run ID
            approved: Whether approved
            approver: Who approved/rejected
            reason: Reason for decision
            checks_passed: Dictionary of approval checks
        """
        event_data = {
            'model_name': model_name,
            'challenger_run_id': challenger_run_id,
            'champion_run_id': champion_run_id,
            'approved': approved,
            'approver': approver,
            'reason': reason,
            'checks_passed': checks_passed,
            'timestamp': datetime.now().isoformat()
        }

        event_hash = self._calculate_event_hash(event_data)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO approval_events
            (model_name, challenger_run_id, champion_run_id, approved, approver, reason, checks_passed, timestamp, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            challenger_run_id,
            champion_run_id,
            1 if approved else 0,
            approver,
            reason,
            json.dumps(checks_passed),
            event_data['timestamp'],
            event_hash
        ))

        conn.commit()
        conn.close()

        status = "approved" if approved else "rejected"
        logger.info(f"✓ Logged model approval: {model_name} {status}")

    def log_drift_alert(
        self,
        model_name: str,
        drift_type: str,
        drift_detected: bool,
        drift_score: float,
        threshold: float,
        action_taken: str = None
    ):
        """
        Log drift detection alert.

        Args:
            model_name: Name of the model
            drift_type: Type of drift ('dataset', 'prediction', 'target')
            drift_detected: Whether drift was detected
            drift_score: Drift score value
            threshold: Threshold for detection
            action_taken: Action taken in response
        """
        event_data = {
            'model_name': model_name,
            'drift_type': drift_type,
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'threshold': threshold,
            'action_taken': action_taken,
            'timestamp': datetime.now().isoformat()
        }

        event_hash = self._calculate_event_hash(event_data)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO drift_events
            (model_name, drift_type, drift_detected, drift_score, threshold, action_taken, timestamp, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            drift_type,
            1 if drift_detected else 0,
            drift_score,
            threshold,
            action_taken,
            event_data['timestamp'],
            event_hash
        ))

        conn.commit()
        conn.close()

        logger.info(f"✓ Logged drift alert: {model_name} ({drift_type})")

    def log_data_access(
        self,
        dataset_name: str,
        user: str,
        access_method: str,
        records_accessed: int = None,
        purpose: str = None
    ):
        """
        Log data access event (for compliance).

        Args:
            dataset_name: Name of dataset accessed
            user: User who accessed data
            access_method: How data was accessed (training, inference, etc.)
            records_accessed: Number of records accessed
            purpose: Purpose of access
        """
        event_data = {
            'event_type': 'data_access',
            'dataset_name': dataset_name,
            'user': user,
            'access_method': access_method,
            'records_accessed': records_accessed,
            'purpose': purpose,
            'timestamp': datetime.now().isoformat()
        }

        event_hash = self._calculate_event_hash(event_data)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO data_access_events
            (event_type, dataset_name, user, access_method, records_accessed, purpose, timestamp, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'data_access',
            dataset_name,
            user,
            access_method,
            records_accessed,
            purpose,
            event_data['timestamp'],
            event_hash
        ))

        conn.commit()
        conn.close()

        logger.debug(f"Logged data access: {dataset_name} by {user}")

    def generate_audit_report(
        self,
        start_date: str = None,
        end_date: str = None,
        event_type: str = None,
        model_name: str = None
    ) -> Dict:
        """
        Generate audit report for specified criteria.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            event_type: Filter by event type
            model_name: Filter by model name

        Returns:
            dict: Audit report with events
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        query = "SELECT * FROM model_lifecycle_events WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)
        events = cursor.fetchall()

        conn.close()

        report = {
            'report_generated': datetime.now().isoformat(),
            'filters': {
                'start_date': start_date,
                'end_date': end_date,
                'event_type': event_type,
                'model_name': model_name
            },
            'total_events': len(events),
            'events': [
                {
                    'id': event[0],
                    'event_type': event[1],
                    'model_name': event[2],
                    'model_version': event[3],
                    'run_id': event[4],
                    'user': event[5],
                    'details': json.loads(event[6]) if event[6] else {},
                    'timestamp': event[7],
                    'event_hash': event[8]
                }
                for event in events
            ]
        }

        logger.info(f"✓ Generated audit report: {len(events)} events")

        return report

    def verify_audit_integrity(self) -> Dict:
        """
        Verify integrity of audit log by checking hashes.

        Returns:
            dict: Verification results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM model_lifecycle_events")
        events = cursor.fetchall()

        conn.close()

        total = len(events)
        verified = 0
        corrupted = []

        for event in events:
            event_data = {
                'event_type': event[1],
                'model_name': event[2],
                'run_id': event[4],
                'user': event[5],
                'details': json.loads(event[6]) if event[6] else {},
                'timestamp': event[7]
            }

            calculated_hash = self._calculate_event_hash(event_data)
            stored_hash = event[8]

            if calculated_hash == stored_hash:
                verified += 1
            else:
                corrupted.append(event[0])

        results = {
            'total_events': total,
            'verified': verified,
            'corrupted': len(corrupted),
            'integrity_status': 'PASS' if len(corrupted) == 0 else 'FAIL',
            'corrupted_ids': corrupted
        }

        if results['integrity_status'] == 'PASS':
            logger.info(f"✓ Audit log integrity verified: {verified}/{total} events")
        else:
            logger.error(f"✗ Audit log integrity compromised: {len(corrupted)} corrupted events")

        return results


def main():
    """Demo audit logging."""
    logger.info("Audit Logger Demo")

    audit = AuditLogger()

    # Log various events
    audit.log_model_training(
        model_name="heart_disease_classifier",
        run_id="demo_run_123",
        user="data_scientist_1",
        details={'auc': 0.85}
    )

    audit.log_model_promotion(
        model_name="heart_disease_classifier",
        model_version="1",
        from_alias="challenger",
        to_alias="champion",
        user="ml_engineer_1",
        reason="Performance improvement confirmed"
    )

    audit.log_drift_alert(
        model_name="heart_disease_classifier",
        drift_type="dataset",
        drift_detected=True,
        drift_score=0.15,
        threshold=0.1,
        action_taken="Model retraining initiated"
    )

    # Generate report
    report = audit.generate_audit_report(model_name="heart_disease_classifier")
    print(f"\nAudit Report: {report['total_events']} events found")

    # Verify integrity
    verification = audit.verify_audit_integrity()
    print(f"Integrity: {verification['integrity_status']} ({verification['verified']}/{verification['total_events']})")


if __name__ == '__main__':
    main()
