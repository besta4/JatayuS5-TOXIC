"""
auth/rbac.py — Role-Based Access Control for Jatayu.

Defines permissions for each user type and provides
decorators/dependencies for route protection.
"""

from enum import Enum
from typing import Set


class Permission(str, Enum):
    """Available permissions in the system."""
    # User self-service
    VIEW_OWN_PROFILE = "view_own_profile"
    UPDATE_OWN_PROFILE = "update_own_profile"
    VIEW_OWN_ACCOUNTS = "view_own_accounts"
    VIEW_OWN_TRANSACTIONS = "view_own_transactions"

    # Transaction permissions
    CREATE_TRANSACTION = "create_transaction"
    RECEIVE_PAYMENTS = "receive_payments"
    MANAGE_PAYEES = "manage_payees"

    # Merchant permissions
    VIEW_PAYMENT_ANALYTICS = "view_payment_analytics"
    VIEW_SETTLEMENT_REPORTS = "view_settlement_reports"

    # Admin permissions
    VIEW_ALL_USERS = "view_all_users"
    VIEW_ALL_TRANSACTIONS = "view_all_transactions"
    VIEW_FRAUD_DASHBOARD = "view_fraud_dashboard"
    MANAGE_USER_STATUS = "manage_user_status"
    REVIEW_HELD_TRANSACTIONS = "review_held_transactions"
    APPROVE_TRANSACTIONS = "approve_transactions"
    BLOCK_TRANSACTIONS = "block_transactions"
    VIEW_COMPLIANCE_REPORTS = "view_compliance_reports"
    GENERATE_STR = "generate_str"
    VIEW_SYSTEM_METRICS = "view_system_metrics"
    MANAGE_LIMITS = "manage_limits"


# Permission sets for each role
ROLE_PERMISSIONS: dict[str, Set[Permission]] = {
    "CUSTOMER": {
        Permission.VIEW_OWN_PROFILE,
        Permission.UPDATE_OWN_PROFILE,
        Permission.VIEW_OWN_ACCOUNTS,
        Permission.VIEW_OWN_TRANSACTIONS,
        Permission.CREATE_TRANSACTION,
        Permission.RECEIVE_PAYMENTS,
        Permission.MANAGE_PAYEES,
    },
    "MERCHANT": {
        Permission.VIEW_OWN_PROFILE,
        Permission.UPDATE_OWN_PROFILE,
        Permission.VIEW_OWN_ACCOUNTS,
        Permission.VIEW_OWN_TRANSACTIONS,
        Permission.RECEIVE_PAYMENTS,
        Permission.VIEW_PAYMENT_ANALYTICS,
        Permission.VIEW_SETTLEMENT_REPORTS,
    },
    "ADMIN": {
        # Admins have all permissions
        Permission.VIEW_OWN_PROFILE,
        Permission.UPDATE_OWN_PROFILE,
        Permission.VIEW_OWN_ACCOUNTS,
        Permission.VIEW_OWN_TRANSACTIONS,
        Permission.VIEW_ALL_USERS,
        Permission.VIEW_ALL_TRANSACTIONS,
        Permission.VIEW_FRAUD_DASHBOARD,
        Permission.MANAGE_USER_STATUS,
        Permission.REVIEW_HELD_TRANSACTIONS,
        Permission.APPROVE_TRANSACTIONS,
        Permission.BLOCK_TRANSACTIONS,
        Permission.VIEW_COMPLIANCE_REPORTS,
        Permission.GENERATE_STR,
        Permission.VIEW_SYSTEM_METRICS,
        Permission.MANAGE_LIMITS,
        Permission.VIEW_PAYMENT_ANALYTICS,
    },
}


def has_permission(user_type: str, permission: Permission) -> bool:
    """Check if a user type has a specific permission."""
    permissions = ROLE_PERMISSIONS.get(user_type, set())
    return permission in permissions


def get_permissions(user_type: str) -> Set[Permission]:
    """Get all permissions for a user type."""
    return ROLE_PERMISSIONS.get(user_type, set())


def can_access_route(user_type: str, required_roles: list[str]) -> bool:
    """
    Check if user type can access a route that requires specific roles.

    Args:
        user_type: The user's type (CUSTOMER, MERCHANT, ADMIN)
        required_roles: List of roles that can access the route

    Returns:
        True if user can access, False otherwise
    """
    # ADMIN can access everything
    if user_type == "ADMIN":
        return True

    return user_type in required_roles
