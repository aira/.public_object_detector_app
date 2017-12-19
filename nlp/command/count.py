from nlp.dispatch import Dispatchable
from nlp.core import pluralize, describe_scene
import typing
import object_detection.constants as const


class CountObjects(Dispatchable):

    def __init__(self, state_q):
        self.state_q = state_q

    def __call__(self, payload):
        state = self.state_q.get()

        vec = const.to_object_series_list(state)

        categories = set([obj['category'] for obj in vec])

        target = self._get_target_object(payload, categories)

        if target is not None:
            response = describe_scene(state,
                                      include_categories=[target],
                                      include_position=False,
                                      include_color=False)
        else:
            response = describe_scene(state,
                                      include_position=False,
                                      include_color=False)
        payload = {
            'response': response
        }

        self.send(payload)

    @staticmethod
    def _get_target_object(payload, categories) -> typing.Optional[str]:
        command = payload['command']

        possible_targets = [(cat, pluralize(cat)) for cat in categories]

        for cat, plcat in possible_targets:
            if cat in command or plcat in command:
                return cat

        return None




