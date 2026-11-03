using System;
using System.Collections.Generic;

// Wire types for /ws/{conversation_id}. Field names and types mirror
// docs/client-contract.schema.json; tools/check_unity_schema.py enforces it.
// Enum-valued strings carry their allowed values in an "enum:" comment. A field that is null on the wire
// is marked "nullable" (empty string here: JsonUtility cannot tell them apart). Defaults mirror the schema's.
// One field per line, public, no attributes: the checker rejects anything it cannot read.
namespace ContextLens
{
    [Serializable]
    public class Turn
    {
        public string id;
        public string conversation_id;
        public int seq;
        public string speaker;
        public float speaker_confidence = 1f;
        public string text;
        public int start_ms;
        public int end_ms;
        public string source; // enum: dataset|synthetic|asr
    }

    [Serializable]
    public class EvidenceRef
    {
        public string turn_id;
        public string quote;
        public string speaker;
        public int start_ms;
    }

    [Serializable]
    public class Finding
    {
        public string agent;
        public string kind; // enum: recall|consistency|context
        public string statement;
        public List<EvidenceRef> evidence;
        public string label; // enum: consistent|update|contradiction|insufficient; nullable
        public float confidence;
    }

    [Serializable]
    public class Card
    {
        public string mode; // enum: recall|clarify|respond
        public string status; // enum: answered|partial|abstained
        public List<Finding> findings;
        public string suggestion; // nullable
        public List<string> missing;
        public int latency_ms;
        public int tokens;
        public float cost_usd;
    }

    [Serializable]
    public class ButtonRequest
    {
        public string conversation_id;
        public int as_of_seq;
        public string question; // nullable
        public string detail = "standard"; // enum: brief|standard|detailed
    }

    [Serializable]
    public class AudioMessage
    {
        public string type = "audio";
        public string data;
        public int sample_rate = 16000;
    }

    [Serializable]
    public class PressMessage
    {
        public string type = "press";
        public ButtonRequest request;
    }

    [Serializable]
    public class PauseMessage
    {
        public string type = "pause";
    }

    [Serializable]
    public class ResumeMessage
    {
        public string type = "resume";
    }

    [Serializable]
    public class DeleteMessage
    {
        public string type = "delete";
    }

    [Serializable]
    public class TranscriptMessage
    {
        public string type = "transcript";
        public Turn turn;
    }

    [Serializable]
    public class CardMessage
    {
        public string type = "card";
        public Card card;
    }

    [Serializable]
    public class StatusMessage
    {
        public string type = "status";
        public string state; // enum: recording|paused|deleted|error
        public string detail; // nullable
    }

    // Used only to read "type" before choosing the full message class. Not part of the contract.
    [Serializable]
    public class Envelope
    {
        public string type;
    }
}
