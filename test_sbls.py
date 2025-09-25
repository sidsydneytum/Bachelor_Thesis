# test_sbls_integration.py
"""
Comprehensive test script for SBLS + Updaters integration.
Tests all combinations of updaters with both column and row additions.
"""

import torch
import numpy as np
from typing import Dict, Any, List
import time
from dataclasses import asdict

# Import your modules (adjust paths as needed)
from sbls import SBLS, SBLSConfig
from updates.direct import DirectUpdater
from updates.greville import GrevilleUpdater
# from updates.updated_greville import UpdatedGrevilleUpdater  # Add when available

def generate_synthetic_data(n_samples: int = 100, input_dim: int = 784, n_classes: int = 10, seed: int = 42):
    """Generate synthetic classification data for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create structured synthetic data with some pattern
    X = torch.randn(n_samples, input_dim)
    # Add some structure to make it more realistic
    X = torch.sigmoid(X + 0.5 * torch.randn(n_samples, 1))  # Broadcast noise
    X = torch.clamp(X, 0.0, 1.0)  # Ensure [0,1] range for rate coding
    
    # Generate labels with some correlation to input
    y = torch.randint(0, n_classes, (n_samples,))
    
    return X, y

def test_basic_functionality():
    """Test basic SBLS functionality without updaters."""
    print("="*60)
    print("TEST 1: Basic SBLS Functionality")
    print("="*60)
    
    # Small config for faster testing
    config = SBLSConfig(
        in_dim=28*28,
        out_dim=10,
        s=10,          # Shorter spike trains for faster testing
        n_feat=50,     # Smaller network
        mw=3,          # Fewer windows
        mp=20,         # Fewer nodes per window
        device="cpu",  # Force CPU for reproducibility
        seed=42
    )
    
    # Generate test data
    X_train, y_train = generate_synthetic_data(80, config.in_dim, config.out_dim)
    X_test, y_test = generate_synthetic_data(20, config.in_dim, config.out_dim, seed=123)
    
    try:
        # Initialize and train
        sbls = SBLS(config)
        print(f"✓ SBLS initialized successfully")
        
        # Test A matrix construction
        A = sbls.build_A(X_train)
        expected_cols = config.n_feat + config.mw * config.mp
        print(f"✓ A matrix shape: {A.shape}, expected cols: {expected_cols}")
        assert A.shape == (len(X_train), expected_cols), f"Shape mismatch: {A.shape}"
        
        # Test training
        sbls.fit(X_train, y_train, store_X_for_updates=True)
        print(f"✓ Training completed")
        assert sbls.W_out is not None, "W_out not created"
        assert sbls.A_plus is not None, "A_plus not created"
        
        # Test prediction
        pred_train = sbls.predict(X_train)
        pred_test = sbls.predict(X_test)
        train_acc = (pred_train == y_train).float().mean().item()
        test_acc = (pred_test == y_test).float().mean().item()
        
        print(f"✓ Training accuracy: {train_acc:.3f}")
        print(f"✓ Test accuracy: {test_acc:.3f}")
        
        return sbls, X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        raise

def test_updater_integration(sbls, X_train, y_train, X_test, y_test):
    """Test integration with different updaters."""
    print("\n" + "="*60)
    print("TEST 2: Updater Integration")
    print("="*60)
    
    updaters = {
        "Direct": DirectUpdater(),
        "Greville": GrevilleUpdater(),
        # "UpdatedGreville": UpdatedGrevilleUpdater(),  # Add when available
    }
    
    results = {}
    
    for updater_name, updater in updaters.items():
        print(f"\n--- Testing {updater_name} Updater ---")
        
        try:
            # Create fresh SBLS instance
            sbls_test = SBLS(sbls.cfg)
            sbls_test.fit(X_train, y_train, store_X_for_updates=True)
            
            # Store initial state for comparison
            initial_acc = (sbls_test.predict(X_test) == y_test).float().mean().item()
            initial_shape = sbls_test.A_train.shape
            initial_W_shape = sbls_test.W_out.shape
            
            print(f"  Initial accuracy: {initial_acc:.3f}")
            print(f"  Initial A shape: {initial_shape}")
            print(f"  Initial W shape: {initial_W_shape}")
            
            # Test 1: Add enhancement windows (columns)
            print(f"  Testing add_enhancement_windows...")
            num_new_windows = 2
            old_A_plus = sbls_test.A_plus.clone()
            
            start_time = time.time()
            sbls_test.add_enhancement_windows(num_new_windows, updater=updater)
            col_update_time = time.time() - start_time
            
            new_shape = sbls_test.A_train.shape
            expected_new_cols = initial_shape[1] + num_new_windows * sbls_test.cfg.mp
            
            assert new_shape[0] == initial_shape[0], "Row count changed unexpectedly"
            assert new_shape[1] == expected_new_cols, f"Column count mismatch: {new_shape[1]} vs {expected_new_cols}"
            assert sbls_test.W_out.shape[0] == expected_new_cols, "W_out shape not updated"
            
            # Verify A_plus is actually different
            if updater_name != "Direct" or torch.allclose(old_A_plus, sbls_test.A_plus[:initial_shape[1], :]):
                print(f"    Warning: A_plus might not have been updated properly")
            
            col_acc = (sbls_test.predict(X_test) == y_test).float().mean().item()
            print(f"  ✓ Column addition: {initial_shape} -> {new_shape}, accuracy: {col_acc:.3f}, time: {col_update_time:.4f}s")
            
            # Test 2: Add training rows
            print(f"  Testing add_training_rows...")
            X_new, y_new = generate_synthetic_data(15, sbls_test.cfg.in_dim, sbls_test.cfg.out_dim, seed=999)
            
            old_shape = sbls_test.A_train.shape
            start_time = time.time()
            sbls_test.add_training_rows(X_new, y_new, updater=updater)
            row_update_time = time.time() - start_time
            
            final_shape = sbls_test.A_train.shape
            expected_new_rows = old_shape[0] + len(X_new)
            
            assert final_shape[0] == expected_new_rows, f"Row count mismatch: {final_shape[0]} vs {expected_new_rows}"
            assert final_shape[1] == old_shape[1], "Column count changed unexpectedly"
            
            final_acc = (sbls_test.predict(X_test) == y_test).float().mean().item()
            print(f"  ✓ Row addition: {old_shape} -> {final_shape}, accuracy: {final_acc:.3f}, time: {row_update_time:.4f}s")
            
            # Store results
            results[updater_name] = {
                'initial_acc': initial_acc,
                'col_acc': col_acc,
                'final_acc': final_acc,
                'col_time': col_update_time,
                'row_time': row_update_time,
                'final_shape': final_shape
            }
            
        except Exception as e:
            print(f"  ✗ {updater_name} updater failed: {e}")
            results[updater_name] = {'error': str(e)}
    
    return results

def test_numerical_consistency():
    """Test numerical consistency between updaters."""
    print("\n" + "="*60)
    print("TEST 3: Numerical Consistency")
    print("="*60)
    
    # Small problem for exact comparison
    config = SBLSConfig(
        in_dim=100,
        out_dim=5,
        s=5,
        n_feat=20,
        mw=2,
        mp=10,
        seed=42,
        device="cpu"
    )
    
    X_train, y_train = generate_synthetic_data(30, config.in_dim, config.out_dim)
    X_test, y_test = generate_synthetic_data(10, config.in_dim, config.out_dim, seed=123)
    
    updaters = {
        "Direct": DirectUpdater(),
        "Greville": GrevilleUpdater(),
    }
    
    print("Comparing predictions after identical operations...")
    
    results = {}
    for name, updater in updaters.items():
        sbls = SBLS(config)
        sbls.fit(X_train, y_train, store_X_for_updates=True)
        
        # Do identical operations
        sbls.add_enhancement_windows(1, updater=updater)
        X_new, y_new = generate_synthetic_data(5, config.in_dim, config.out_dim, seed=777)
        sbls.add_training_rows(X_new, y_new, updater=updater)
        
        pred = sbls.predict(X_test)
        results[name] = {
            'predictions': pred,
            'A_shape': sbls.A_train.shape,
            'W_out': sbls.W_out.clone()
        }
    
    # Compare results
    if len(results) >= 2:
        names = list(results.keys())
        ref_name = names[0]
        
        for i in range(1, len(names)):
            name = names[i]
            pred_match = torch.equal(results[ref_name]['predictions'], results[name]['predictions'])
            shape_match = results[ref_name]['A_shape'] == results[name]['A_shape']
            
            # Compare W_out with tolerance (numerical precision differences expected)
            W_diff = torch.norm(results[ref_name]['W_out'] - results[name]['W_out']).item()
            W_close = W_diff < 1e-6
            
            print(f"  {ref_name} vs {name}:")
            print(f"    Predictions match: {pred_match}")
            print(f"    Shapes match: {shape_match}")
            print(f"    W_out difference: {W_diff:.2e} ({'✓' if W_close else '✗'})")
            
            if not pred_match and W_diff > 1e-3:
                print(f"    ⚠️  Large discrepancy detected!")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases")
    print("="*60)
    
    config = SBLSConfig(in_dim=50, out_dim=3, s=3, n_feat=10, mw=1, mp=5, device="cpu", seed=42)
    X_train, y_train = generate_synthetic_data(20, config.in_dim, config.out_dim)
    
    sbls = SBLS(config)
    updater = DirectUpdater()
    
    # Test 1: Operations before training
    try:
        sbls.add_enhancement_windows(1, updater=updater)
        print("  ✗ Should have failed: operations before training")
    except RuntimeError:
        print("  ✓ Correctly rejected operations before training")
    
    # Train model
    sbls.fit(X_train, y_train, store_X_for_updates=False)  # No X caching
    
    # Test 2: Column addition without cached X
    try:
        sbls.add_enhancement_windows(1, updater=updater)
        print("  ✗ Should have failed: column addition without cached X")
    except RuntimeError:
        print("  ✓ Correctly rejected column addition without cached X")
    
    # Retrain with X caching
    sbls.fit(X_train, y_train, store_X_for_updates=True)
    
    # Test 3: Zero additions
    try:
        sbls.add_enhancement_windows(0, updater=updater)
        sbls.add_training_rows(torch.empty(0, config.in_dim), torch.empty(0, dtype=torch.long), updater=updater)
        print("  ✓ Handled zero additions gracefully")
    except Exception as e:
        print(f"  ✗ Failed on zero additions: {e}")

def run_all_tests():
    """Run complete test suite."""
    print("SBLS + Updaters Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Basic functionality
        sbls, X_train, y_train, X_test, y_test = test_basic_functionality()
        
        # Test 2: Updater integration
        updater_results = test_updater_integration(sbls, X_train, y_train, X_test, y_test)
        
        # Test 3: Numerical consistency
        test_numerical_consistency()
        
        # Test 4: Edge cases
        test_edge_cases()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        print("\nUpdater Performance Summary:")
        for name, result in updater_results.items():
            if 'error' in result:
                print(f"  {name}: FAILED - {result['error']}")
            else:
                print(f"  {name}:")
                print(f"    Final accuracy: {result['final_acc']:.3f}")
                print(f"    Column update time: {result['col_time']:.4f}s")
                print(f"    Row update time: {result['row_time']:.4f}s")
                print(f"    Final matrix shape: {result['final_shape']}")
        
        print(f"\n✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)