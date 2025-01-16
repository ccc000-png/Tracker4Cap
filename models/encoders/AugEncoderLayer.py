import torch
from torch import nn


class AugEncoder(nn.Module):
    def __init__(self, ObjectEncoder, ActionEncoder, max_objects, visual_dim, object_dim,
                                           hidden_dim):
        super(AugEncoder, self).__init__()
        self.video_embeddings = nn.Linear(visual_dim, hidden_dim)
        if object_dim is not None:
            self.object_embeddings = nn.Linear(object_dim, hidden_dim)
        self.track_objects = max_objects
        self.object_track = ObjectEncoder
        self.action_track = ActionEncoder

    def forward(self, visual, objects = None, query_pos = None):
        vhidden_states = self.video_embeddings(visual)

        if objects is not None:
            ohidden_states = self.object_embeddings(objects)
        if self.object_track is not None:
            object_hidden_states= self.object_track(vhidden_states, query_pos, self.track_objects, mask=None)
        else:
            object_hidden_states = vhidden_states
        if self.action_track is not None:
            action_features = self.action_track(object_hidden_states)
        else:
            action_features = object_hidden_states

        return object_hidden_states, action_features