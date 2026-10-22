"""Clinician step-by-step plan workflow sessions.

Revision ID: 0006_plan_sessions
Revises: 0005_pii_identity
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_plan_sessions"
down_revision = "0005_pii_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    if not insp.has_table("plan_sessions"):
        op.create_table(
            "plan_sessions",
            sa.Column("id", sa.String(40), primary_key=True),
            sa.Column("case_id", sa.String(64), nullable=False, index=True),
            sa.Column("status", sa.String(16), nullable=False, server_default="active"),
            sa.Column("current_step", sa.String(24), nullable=False, server_default="intake"),
            sa.Column("state", sa.JSON, nullable=False),
            sa.Column("owner_clinician_id", sa.String(64)),
            sa.Column("clinic_id", sa.String(64)),
            sa.Column("report_id", sa.String(40)),
            sa.Column("created_at", sa.DateTime),
            sa.Column("updated_at", sa.DateTime),
        )


def downgrade() -> None:
    try:
        op.drop_table("plan_sessions")
    except Exception:  # noqa: BLE001
        pass
