PYTHON ?= python3
BOOTSTRAP := $(PYTHON) scripts/bootstrap.py

.PHONY: reproduce reproduce-full realdata-smoke realdata-nested-160 realdata-switching-160 realdata-research-320 robustness-evaluation posthoc-methods test notebook kaggle-bundle verify clean-evidence

reproduce:
	$(BOOTSTRAP) -m pytest
	$(BOOTSTRAP) scripts/run_reproduce.py --config configs/smoke.json
	$(BOOTSTRAP) scripts/run_deployment_smoke.py
	$(BOOTSTRAP) scripts/build_portfolio_notebook.py
	$(BOOTSTRAP) scripts/execute_portfolio_notebook.py
	$(BOOTSTRAP) scripts/qa_math_render.py
	$(BOOTSTRAP) scripts/verify_reproduction.py

reproduce-full:
	@test -n "$(DATA_ROOT)" || (echo "Set DATA_ROOT=/path/to/rogii-wellbore-geology-prediction.zip" && exit 2)
	$(BOOTSTRAP) scripts/run_reproduce.py --config configs/full.json --data-root "$(DATA_ROOT)"
	$(BOOTSTRAP) scripts/build_portfolio_notebook.py
	$(BOOTSTRAP) scripts/execute_portfolio_notebook.py
	$(BOOTSTRAP) scripts/qa_math_render.py
	$(BOOTSTRAP) scripts/verify_reproduction.py

realdata-smoke:
	@test -n "$(DATA_ROOT)" || (echo "Set DATA_ROOT=/path/to/raw-data-directory-or-zip" && exit 2)
	$(BOOTSTRAP) scripts/run_reproduce.py --config configs/realdata_smoke.json --data-root "$(DATA_ROOT)" --output-root artifacts/realdata_smoke

realdata-nested-160:
	@test -n "$(DATA_ROOT)" || (echo "Set DATA_ROOT=/path/to/raw-data-directory-or-zip" && exit 2)
	$(BOOTSTRAP) scripts/run_reproduce.py --config configs/realdata_nested_160.json --data-root "$(DATA_ROOT)" --output-root artifacts/realdata_nested_160

realdata-switching-160:
	@test -n "$(DATA_ROOT)" || (echo "Set DATA_ROOT=/path/to/raw-data-directory-or-zip" && exit 2)
	$(BOOTSTRAP) scripts/run_reproduce.py --config configs/realdata_switching_160.json --data-root "$(DATA_ROOT)" --output-root artifacts/realdata_switching_160

realdata-research-320:
	@test -n "$(DATA_ROOT)" || (echo "Set DATA_ROOT=/path/to/raw-data-directory-or-zip" && exit 2)
	$(BOOTSTRAP) scripts/run_reproduce.py --config configs/realdata_nested_160.json --data-root "$(DATA_ROOT)" --output-root artifacts/realdata_nested_160
	$(BOOTSTRAP) scripts/audit_component_graph.py --config configs/realdata_nested_160.json --data-root "$(DATA_ROOT)" --artifact-root artifacts/realdata_nested_160
	$(BOOTSTRAP) scripts/run_regret_router.py --artifact-root artifacts/realdata_nested_160
	$(BOOTSTRAP) scripts/fit_primary_policy.py --artifact-root artifacts/realdata_nested_160
	$(BOOTSTRAP) scripts/run_reproduce.py --config configs/realdata_nested_160_confirmation.json --data-root "$(DATA_ROOT)" --output-root artifacts/realdata_nested_160_confirmation
	$(BOOTSTRAP) scripts/evaluate_frozen_policy.py --policy artifacts/realdata_nested_160/frozen_primary_policy.json --confirmation-root artifacts/realdata_nested_160_confirmation
	$(BOOTSTRAP) scripts/evaluate_meta_shrinkage.py --primary-root artifacts/realdata_nested_160 --confirmation-root artifacts/realdata_nested_160_confirmation

robustness-evaluation:
	$(BOOTSTRAP) scripts/evaluate_robustness.py

posthoc-methods:
	$(BOOTSTRAP) scripts/experiment_group_robust_stack.py
	$(BOOTSTRAP) scripts/experiment_trust_region_ridge.py

test:
	$(BOOTSTRAP) -m pytest

notebook:
	$(BOOTSTRAP) scripts/build_portfolio_notebook.py
	$(BOOTSTRAP) scripts/execute_portfolio_notebook.py
	$(BOOTSTRAP) scripts/qa_math_render.py

kaggle-bundle:
	$(BOOTSTRAP) scripts/build_kaggle_companion.py

verify:
	$(BOOTSTRAP) scripts/verify_reproduction.py

clean-evidence:
	@echo "Evidence is intentionally not deleted automatically. Remove generated evidence manually if required."
