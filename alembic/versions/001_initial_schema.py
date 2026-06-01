"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-06-01

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chroma_collection", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_documents_created_at", "documents", ["created_at"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("contexts", postgresql.JSONB(), nullable=False),
        sa.Column("faithfulness", sa.Float(), nullable=True),
        sa.Column("answer_relevancy", sa.Float(), nullable=True),
        sa.Column("context_precision", sa.Float(), nullable=True),
        sa.Column("context_recall", sa.Float(), nullable=True),
        sa.Column("is_hallucination", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column("ground_truth", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_evaluation_runs_created_at", "evaluation_runs", ["created_at"])
    op.create_index(
        "ix_evaluation_runs_document_created",
        "evaluation_runs",
        ["document_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_evaluation_runs_document_created", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_created_at", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
    op.drop_index("ix_documents_created_at", table_name="documents")
    op.drop_table("documents")
