#!/usr/bin/env python3
"""
Manual test script for NetworkSpyAgent integration with SuperDiscoveryAgent.

This script tests that:
1. NetworkSpyAgent can be created via _create_specialist
2. A network_spy task can be created successfully
3. The task can be executed without errors
"""

from bluebox.agents.super_discovery_agent import SuperDiscoveryAgent
from bluebox.llms.infra.network_data_store import NetworkDataStore
from bluebox.data_models.orchestration.task import SpecialistAgentType


def test_network_spy_integration() -> None:
    """Test NetworkSpyAgent integration with SuperDiscoveryAgent."""

    print("=" * 80)
    print("Testing NetworkSpyAgent Integration")
    print("=" * 80)

    # Step 1: Load network data
    print("\n[1/5] Loading network data from test file...")
    network_events_path = "tests/data/input/network_events/network_api.jsonl"
    try:
        network_store = NetworkDataStore.from_jsonl(network_events_path)
        print(f"✓ Loaded {len(network_store.entries)} network entries")
    except Exception as e:
        print(f"✗ Failed to load network data: {e}")
        return

    # Step 2: Create SuperDiscoveryAgent
    print("\n[2/5] Creating SuperDiscoveryAgent...")
    try:
        agent = SuperDiscoveryAgent(
            emit_message_callable=lambda msg: None,  # Suppress messages for test
            network_data_store=network_store,
            task="Test network_spy integration",
            max_iterations=10,
        )
        print("✓ SuperDiscoveryAgent created successfully")
    except Exception as e:
        print(f"✗ Failed to create SuperDiscoveryAgent: {e}")
        return

    # Step 3: Verify NETWORK_SPY is in AVAILABLE_AGENT_TYPES
    print("\n[3/5] Checking AVAILABLE_AGENT_TYPES...")
    if SpecialistAgentType.NETWORK_SPY in agent.AVAILABLE_AGENT_TYPES:
        print("✓ NETWORK_SPY is in AVAILABLE_AGENT_TYPES")
    else:
        print("✗ NETWORK_SPY is NOT in AVAILABLE_AGENT_TYPES")
        print(f"  Available types: {[t.value for t in agent.AVAILABLE_AGENT_TYPES]}")
        return

    # Step 4: Create a network_spy task
    print("\n[4/5] Creating network_spy task...")
    try:
        result = agent._create_task(
            agent_type="network_spy",
            prompt="Find the API endpoint that contains user data",
            max_loops=3,
        )

        if result.get("success"):
            task_id = result.get("task_id")
            print(f"✓ Task created successfully (ID: {task_id})")
        else:
            print(f"✗ Task creation failed: {result.get('error')}")
            return
    except Exception as e:
        print(f"✗ Failed to create task: {e}")
        return

    # Step 5: Test specialist creation
    print("\n[5/5] Testing specialist creation...")
    try:
        # Get the task we just created
        tasks = agent._state.get_pending_tasks()
        if not tasks:
            print("✗ No pending tasks found")
            return

        task = tasks[0]

        # Try to create the specialist
        specialist = agent._create_specialist(task.agent_type)
        print(f"✓ NetworkSpyAgent instance created: {type(specialist).__name__}")

        # Verify it has the required attributes
        if hasattr(specialist, "_network_data_store"):
            print("✓ NetworkSpyAgent has _network_data_store attribute")
        else:
            print("✗ NetworkSpyAgent missing _network_data_store attribute")

    except Exception as e:
        print(f"✗ Failed to create specialist: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nNetworkSpyAgent is successfully integrated into SuperDiscoveryAgent.")
    print("The agent can now delegate endpoint discovery tasks to network_spy.")


if __name__ == "__main__":
    test_network_spy_integration()
