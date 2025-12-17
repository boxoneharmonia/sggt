# Network Policy Analysis Report

## Objective
The objective of this analysis is to distinguish whether the policy of the network works as intended, specifically focusing on its ability to predict pose and 2D corners.

## Architecture Overview
The network (`MyNet`) employs an encoder-decoder architecture:
-   **Encoder**: A transformer-based backbone (utilizing a pre-trained base, e.g., Dinov2) that extracts hierarchical features and a global camera token.
-   **Decoder**:
    -   **Pose Head**: Uses the global camera token with a `DynamicRouter` to fuse features and regress the 6D rotation and translation directly.
    -   **Corner Head**: Uses a `DynamicRouter` to select encoder features, which are then processed by a `DPTHead` (Dense Prediction Transformer Head). This head outputs a `pvmap` (Pixel-wise Voting Map) representing 2D vectors from pixels to corners, and a `mask`.
    -   **Point Map Head**: Predicts dense 3D point clouds (`pcloud`) and confidence maps (`pconf`).

## Issue Identification
During the code review and unit testing, a critical dimension mismatch was identified:
-   **Configuration**: `config.py` sets `self.maps = 8`.
-   **Network Logic**: The `DPTHead` outputs `config.maps` channels for the voting map. Since each corner requires 2 coordinates (x, y), `maps=8` implies the network predicts **4 corners**.
-   **Dataset**: `pose/dataset.py` provides **8 corners** (likely the 8 vertices of a 3D bounding box) in `pts3d`.
-   **Loss Function**: `PoseLoss.voting_loss` attempts to calculate the distance between the projected 3D ground truth points (8 points) and the predicted 2D corners (4 points).

This mismatch causes a **RuntimeError** during the loss calculation because the tensor shapes are incompatible.

## Verification
A reproduction script (`tests/reproduce_issue.py`) was created to simulate the forward pass and loss calculation using the default configuration and dummy data matching the dataset structure.

**Result**:
```
RuntimeError: The size of tensor a (8) must match the size of tensor b (4) at non-singleton dimension 1
```
This confirms that the network policy **does not work** with the current configuration.

## Proposed Fix
To resolve this, the `config.maps` parameter must be increased to accommodate 8 corners.
-   **Fix**: Set `config.maps = 16` (8 corners * 2 coordinates).

## Validation
A validation script (`tests/verify_fix.py`) was created to test the network with the proposed fix.

**Result**:
-   **Forward Pass**: Successful.
-   **Loss Calculation**: Successful.
-   **Total Loss**: Computed without errors.

## Conclusion
The underlying policy and architecture of the network are sound. The dynamic routing and voting mechanism for corner prediction are implemented correctly in principle. However, the configuration file contained an incorrect parameter (`maps=8`) that was inconsistent with the dataset (8 corners), causing the network to fail.

**Recommendation**: Update `config.py` to set `self.maps = 16`.
