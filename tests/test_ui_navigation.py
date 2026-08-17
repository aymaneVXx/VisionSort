from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from visionsort.ui.components.common import event_label, status_label


EXPECTED_PAGES = [
    "Accueil",
    "Caméras",
    "Configuration du site",
    "Calibration",
    "Exploitation",
    "Historique & alertes",
    "Modèles",
    "Diagnostic",
    "Enregistrements",
    "Données & annotations",
    "Entraînement",
]


def _navigation(app: AppTest):
    return next(radio for radio in app.radio if radio.label == "Navigation")


def test_all_application_pages_render_without_streamlit_exception():
    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert list(_navigation(app).options) == EXPECTED_PAGES
    assert not app.exception
    for page in EXPECTED_PAGES:
        _navigation(app).set_value(page)
        app.run()
        assert not app.exception, f"La page {page!r} ne se charge pas correctement."


def test_home_keeps_primary_operator_actions_visible():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    labels = {button.label for button in app.button}

    assert {
        "Démarrer le système",
        "Arrêter",
        "Configurer le site",
        "Ajouter une caméra",
    } <= labels


def test_internal_states_are_translated_for_the_operator():
    assert status_label("RUNNING") == "En fonctionnement"
    assert status_label("OFFLINE") == "Hors ligne"
    assert event_label("handoff_ambiguous") == "Identification du colis à confirmer"


_DASHBOARD_ACTION_APP = """
import streamlit as st
from types import SimpleNamespace
from visionsort.ui.pages import dashboard

class FakeCalibrationRepository:
    def __init__(self, db):
        pass

    def get_active_profile(self, source_id):
        return object()

dashboard.CalibrationRepository = FakeCalibrationRepository

class FakeRepository:
    def list_sources(self):
        return [{"id": "camera-c1", "status": "OFFLINE"}]

    def list_capture_sessions(self):
        return [{
            "id": "session-1",
            "name": "Production",
            "started_at": __STARTED_AT__,
            "ended_at": None,
        }]

    def list_parcels(self):
        return []

    def list_models(self):
        return [{"id": "model-1", "is_active": 1}]

    def recent_events(self, limit):
        return []

    def enqueue_command(self, command_type, payload):
        st.session_state["queued_command"] = command_type.value
        st.session_state["queued_session"] = payload.get("session_id")

context = SimpleNamespace(
    repo=FakeRepository(),
    db=None,
    config_demo_mode=False,
    config_values={},
)
dashboard.render(context)
st.write(
    f"QUEUED={st.session_state.get('queued_command')}|"
    f"{st.session_state.get('queued_session')}"
)
"""


@pytest.mark.parametrize(
    ("started_at", "button_label", "expected_command"),
    [
        ("None", "Démarrer le système", "START_SESSION"),
        ("1.0", "Arrêter", "STOP_SESSION"),
    ],
)
def test_dashboard_runtime_actions_enqueue_the_expected_command(
    started_at: str,
    button_label: str,
    expected_command: str,
):
    script = _DASHBOARD_ACTION_APP.replace("__STARTED_AT__", started_at)
    app = AppTest.from_string(script, default_timeout=20).run()

    button = next(item for item in app.button if item.label == button_label)
    assert not button.disabled
    button.click()
    app.run()

    assert not app.exception
    assert any(
        item.value == f"QUEUED={expected_command}|session-1"
        for item in app.markdown
    )
