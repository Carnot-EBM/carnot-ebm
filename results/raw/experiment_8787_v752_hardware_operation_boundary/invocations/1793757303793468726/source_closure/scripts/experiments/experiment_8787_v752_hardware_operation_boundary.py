"""REQ-REPORT-8787: expose the sealed evidence reader without loading a model."""

from carnot.reporting.hardware_operation_boundary_8787 import main

raise SystemExit(main())
