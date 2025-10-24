# Model Card: Hospital 30-Day Readmission Risk Prediction

**Model Name:** Hospital Readmission Risk Predictor
**Model Type:** Regression (XGBoost)
**Version:** 1.0.0 (Placeholder)
**Status:** In Development
**Last Updated:** 2025-10-22
**Owner:** Community Hospital Data Science Team

---

## Model Overview

### Intended Use
This model predicts the probability (0-1 score) that a patient will be readmitted to the hospital within 30 days of discharge. It is intended to support:
- Care coordination teams in identifying high-risk patients for follow-up
- Discharge planning to ensure appropriate post-discharge support
- Resource allocation for transitional care programs

### Out-of-Scope Use
- **NOT** for denying care or insurance coverage
- **NOT** for punitive measures against patients or providers
- **NOT** for use outside of Community Hospital without revalidation
- **NOT** for pediatric patients (< 18 years old)

---

## Model Details

### Architecture
- **Algorithm:** XGBoost Regression
- **Objective:** Logistic regression (bounded 0-1 output)
- **Framework:** XGBoost 2.0.3
- **Training Framework:** MLflow + FEAST feature store

### Features (Placeholder - To Be Implemented)
**Demographics:**
- Age, Sex, Insurance Type

**Index Admission:**
- Length of stay, Admission type, Discharge disposition, Number of procedures, Number of diagnoses

**Clinical History:**
- Prior admissions, Emergency visits, Chronic conditions, Medications, Medication changes

**Lab Values:**
- Key lab results from admission

**Social Determinants:**
- Home health services, Social support indicators

**Total Features:** ~25 features (to be finalized)

---

## Training Data

### Source
Synthetic patient data generated for demonstration purposes.

**Note:** This is a placeholder model. Production implementation would use:
- Community Hospital EHR data
- Minimum 2 years of historical admissions
- Validated readmission outcomes (30-day window)

### Data Specifications
- **Training Set Size:** 1,000 patients (placeholder)
- **Time Period:** 12 months (placeholder)
- **Positive Rate:** ~20% readmission rate (typical for hospitals)
- **Data Quality:** Synthetic data for demonstration only

### Exclusions
- Patients who died during index admission
- Patients transferred to another facility
- Obstetric admissions
- Psychiatric primary admissions
- Planned readmissions (e.g., chemotherapy)

---

## Performance Metrics

### Overall Performance (Placeholder)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | TBD | Root Mean Squared Error (lower is better) |
| **MAE** | TBD | Mean Absolute Error (lower is better) |
| **R²** | TBD | Coefficient of determination (higher is better) |
| **MAPE** | TBD | Mean Absolute Percentage Error |

**Note:** Metrics will be populated once model training is implemented.

### Performance by Subgroup (Placeholder)
Analysis of prediction error across demographic groups will be added to ensure equitable performance.

---

## Fairness & Bias Analysis

### Protected Attributes
- **Sex:** Female, Male
- **Age Group:** Young (<45), Middle (45-65), Senior (>65)
- **Insurance Type:** Medicare, Medicaid, Commercial, Self-Pay

### Fairness Criteria
- RMSE should not differ by more than 10% across protected groups
- No systematic over/under-prediction for any demographic group
- Regular monitoring for emerging bias

### Current Status
⚠️ **Bias analysis pending** - will be completed during full implementation

---

## Limitations & Risks

### Known Limitations
1. **Synthetic Data:** Current version uses generated data, not real patient outcomes
2. **Limited Features:** Placeholder feature set may not capture all relevant risk factors
3. **Single Hospital:** Model trained on Community Hospital data only
4. **Temporal Validity:** Requires retraining as patient populations and care practices evolve

### Potential Risks
- **False Negatives:** Missing high-risk patients who don't receive interventions
- **False Positives:** Over-allocating resources to low-risk patients
- **Health Equity:** Risk of bias if certain populations are systematically mispredicted
- **Data Quality:** Predictions degrade with incomplete or incorrect input data

### Mitigation Strategies
- Continuous monitoring of prediction accuracy and fairness
- Regular retraining (quarterly recommended)
- Human oversight required for all clinical decisions
- Transparency with patients about model use

---

## Model Lineage

### Data Lineage
```
EHR System → FEAST Feature Store → Model Training
                                  ↓
                            Model Registry (MLflow)
                                  ↓
                            Production Inference
```

### Training Details
- **Training Date:** TBD
- **Dataset Version:** TBD
- **Feature Store Snapshot:** TBD
- **MLflow Experiment ID:** TBD
- **MLflow Run ID:** TBD

---

## Deployment Information

### Current Status
🟡 **Development** - Not deployed to production

### Deployment Requirements
When ready for production:
- Integration with EHR system for real-time feature extraction
- Online feature serving via FEAST
- Prediction latency < 500ms
- Drift monitoring enabled
- Approval workflow completed

### Monitoring Plan
- **Performance Monitoring:** Weekly RMSE tracking
- **Drift Detection:** Monthly dataset drift checks
- **Bias Monitoring:** Quarterly fairness audits
- **Retraining Trigger:** RMSE degrades >10% or significant drift detected

---

## Ethical Considerations

### Transparency
Patients and clinicians should be informed when readmission predictions influence care decisions.

### Autonomy
Predictions are advisory only. Clinical judgment always takes precedence.

### Equity
Regular fairness audits ensure equitable care across all patient demographics.

### Accountability
Data Science team maintains ownership and monitoring of model performance.

---

## References

### Related Documentation
- [Training Code](train.py) - Model training implementation
- [Simplified Version](train_simple.py) - Working demonstration
- [Configuration](config.yaml) - Model parameters
- [Data Generator](../../utils/data_generator.py) - Synthetic data creation

### External Resources
- Hospital Readmission Reduction Program (HRRP): [CMS.gov](https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp)
- LACE Index: Length of stay, Acuity, Comorbidity, Emergency visits

---

## Contact

**Model Owner:** Community Hospital Data Science Team
**Email:** mlops@communityhospital.example.com
**Last Review Date:** 2025-10-22
**Next Review Date:** TBD (Quarterly reviews recommended)

---

**This model card follows the template from:** `src/governance/templates/model_card_template.md`

**Version History:**
- v1.0.0 (2025-10-22): Initial placeholder model card created
