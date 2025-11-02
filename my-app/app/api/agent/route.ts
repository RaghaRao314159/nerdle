import { NextRequest, NextResponse } from "next/server";

// API route that proxies to Python backend service
const PYTHON_BACKEND_URL = process.env.AGENT_SERVER_URL || "http://localhost:5000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { history, attemptNumber } = body;
    
    // Log the incoming request
    console.log("=".repeat(80));
    console.log("AGENT API - Incoming request:");
    console.log("Attempt number:", attemptNumber);
    console.log("History length:", history?.length || 0);
    if (history && history.length > 0) {
      console.log("Last guess:", history[history.length - 1]?.guess);
      console.log("Last feedback:", history[history.length - 1]?.feedback);
    }
    console.log("=".repeat(80));

    // Call Python backend service
    try {
      const response = await fetch(`${PYTHON_BACKEND_URL}/guess`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          history,
          attemptNumber,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Log the response from Python backend
      console.log("=".repeat(80));
      console.log("AGENT API - Response from Python backend:");
      console.log("Guess:", data.guess);
      console.log("Full response:", JSON.stringify(data, null, 2));
      console.log("=".repeat(80));
      
      return NextResponse.json({
        guess: data.guess,
        source: data.source || "unknown",
        metadata: data.metadata || {},
        prompt: formatPromptForModel(history),
      });
    } catch (backendError: any) {
      // Fallback to simple guess if backend is not available
      console.warn("Backend not available, using fallback:", backendError.message);
      const guess = generateSimpleGuess(history);
      
      return NextResponse.json({
        guess,
        source: "fallback",
        metadata: {
          reason: "backend_unavailable",
          error: backendError.message,
        },
        prompt: formatPromptForModel(history),
        warning: "Using fallback guess generator. Python backend may not be running.",
      });
    }
  } catch (error: any) {
    console.error("Error in agent API:", error);
    return NextResponse.json(
      { error: error.message || "Failed to generate guess" },
      { status: 500 }
    );
  }
}

function formatPromptForModel(history: any[]): string {
  if (history.length === 0) {
    return "This is a nerdle game. Make your first guess of an 8-character equation (e.g., 48-32=16).";
  }

  const lines: string[] = [
    "This is a nerdle game.",
    "Legend: g = green (correct spot), p = purple (wrong spot), b = black (not in equation).",
  ];

  history.forEach((item: any, idx: number) => {
    const feedbackStr = item.feedback
      .map((f: string, i: number) => {
        const char = item.guess[i];
        if (f === "correct") return `${char}g`;
        if (f === "present") return `${char}p`;
        return `${char}b`;
      })
      .join(" ");
    lines.push(`Attempt ${idx + 1}: ${item.guess} | Feedback: ${feedbackStr}`);
  });

  lines.push("Find the correct Nerdle equation. Output only the next 8-character guess, nothing else.");
  return lines.join("\n");
}

// Simple rule-based guess generator (placeholder until Python backend is connected)
function generateSimpleGuess(history: any[]): string {
  if (history.length === 0) {
    return "48-32=16"; // Starting guess
  }

  // Very simple logic: just return a placeholder
  // This will be replaced by actual model inference
  const guesses = ["10+20=30", "25+25=50", "12*3=36", "8*8=64", "15+15=30"];
  const index = Math.min(history.length, guesses.length - 1);
  return guesses[index] || "48-32=16";
}

