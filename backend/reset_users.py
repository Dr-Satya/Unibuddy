# reset_users.py
"""Utility script to wipe all users and their related data.
Keeps university data and vector DB intact.
"""

from src.database import db_manager, UserModel, ConversationModel, MessageModel, AuditLogModel


def main() -> None:
    db = db_manager.get_session()
    try:
        print("⚠️ Deleting all users, conversations, messages, and audit logs...")

        msgs = db.query(MessageModel).delete()
        print(f"Deleted messages: {msgs}")

        convs = db.query(ConversationModel).delete()
        print(f"Deleted conversations: {convs}")

        logs = db.query(AuditLogModel).delete()
        print(f"Deleted audit logs: {logs}")

        users = db.query(UserModel).delete()
        print(f"Deleted users: {users}")

        db.commit()
        print("✅ Reset complete.")
    except Exception as e:
        db.rollback()
        print("❌ Error during reset:", e)
    finally:
        db.close()


if __name__ == "__main__":
    main()
