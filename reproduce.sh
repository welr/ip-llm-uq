#!/bin/bash
# Reruns the analyses on the stored runs in dependency order and rewrites generated/.
set -u
PY="${PYTHON:-python}"
cd "$(dirname "$0")/experiments" || exit 1
for s in cf_integration/provenance_checks cf_integration/mc_letter_scoring cf_integration/continuum_gap cf_integration/cf_results new_problems/criteria_comparison new_problems/readmitted_and_strict audit_p1_p2/audit_registered_strata audit_p1_p2/audit_registered_tie audit_p1_p2/audit_p3_outcomes audit_p1_p2/audit_p3_checks audit_p1_p2/audit_p3 audit_p1_p2/registered_analyses_control audit_p1_p2/registered_analyses audit_p1_p2/audit_registered_form audit_p1_p2/audit_p1 audit_p1_p2/audit_p1_cap audit_p1_p2/audit_p1_claims audit_p1_p2/audit_p1_mistral audit_p1_p2/audit_p2 audit_p1_p2/audit_p2_parser audit_p1_p2/audit_p2_reanalysis audit_p1_p2/audit_p2_strata audit_p1_p2/audit_p4 audit_p1_p2/analyze_regen audit_p1_p2/analyze_matched audit_p1_p2/analyze_p1_placebo audit_p1_p2/analyze_neutral audit_p1_p2/analyze_p4_template audit_p1_p2/regen_check_mistral audit_p1_p2/build_pred2_tolerance cf_integration/rescore_pred4 audit_p1_p2/build_p2_p4_text audit_p1_p2/build_p1_text audit_p1_p2/build_pred3_disclosures cf_integration/build_tables; do
  echo "== $s"
  ( cd "$(dirname "$s")" && "$PY" "$(basename "$s").py" > /dev/null ) || { echo "FAILED $s" >&2; exit 1; }
done
echo "All analyses reran."
