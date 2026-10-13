# ──────────────────────────────────────────────────────────────
# main.py  –  The starting point of the whole application
# ──────────────────────────────────────────────────────────────
# When you run this file, it boots up the Flask web server.
# Think of this as the "on switch" for the entire app.

# We use a "factory" pattern, meaning there is a separate helper
# function (create_app) that actually builds and configures the
# Flask app.  This keeps everything tidy and testable.
from app_factory import create_app

# Call the factory to get a fully configured Flask app object.
app = create_app()

# Only run the web server if we are executing THIS file directly.
# (If another file imports this module, the server should NOT start automatically.)
if __name__ == '__main__':
    # Listen on all network interfaces (0.0.0.0) so Docker and
    # external machines can reach the server, not just localhost.
    # Port 5000 is the default Flask port.
    # debug=True means Flask will auto-reload when you change code
    # and will show detailed error pages.  Turn this off in production!
    app.run(host='0.0.0.0', port=5000, debug=True)
