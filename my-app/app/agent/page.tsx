"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";

const MAX_GUESSES = 6;
const EQUATION_LENGTH = 8;
const ALLOWED_INPUT = new Set<string>(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "="]);

type TileState = "correct" | "present" | "absent";
type DisplayTileState = TileState | "pending" | "empty";
type GameStatus = "playing" | "won" | "lost" | "loading";

interface GameHistory {
  guess: string;
  feedback: TileState[];
}

export default function AgentMode() {
  const [solution, setSolution] = useState<string>("");
  const [guesses, setGuesses] = useState<string[]>([]);
  const [feedback, setFeedback] = useState<TileState[][]>([]);
  const [gameStatus, setGameStatus] = useState<GameStatus>("loading");
  const [message, setMessage] = useState<string | null>(null);
  const [isAutoPlaying, setIsAutoPlaying] = useState<boolean>(false);
  const [gameHistory, setGameHistory] = useState<GameHistory[]>([]);
  const [remainingCandidates, setRemainingCandidates] = useState<string[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<"qwen-grpo" | "gemini">("qwen-grpo");

  // Generate equation pool (same as main page)
  const EQUATION_POOL = useMemo(() => generateEquationPool(), []);

  const resetGame = useCallback(() => {
    setSolution(pickRandomEquation(EQUATION_POOL));
    setGuesses([]);
    setFeedback([]);
    setGameStatus("playing");
    setMessage(null);
    setGameHistory([]);
    setRemainingCandidates([...EQUATION_POOL]);
  }, [EQUATION_POOL]);

  // Reset game when agent changes or on mount
  useEffect(() => {
    resetGame();
  }, [selectedAgent, resetGame]);

  // Auto-play logic
  useEffect(() => {
    if (!isAutoPlaying || gameStatus !== "playing" || guesses.length >= MAX_GUESSES) {
      return;
    }

    let isCancelled = false;

    const makeGuess = async () => {
      if (isCancelled) return;
      try {
        setGameStatus("loading");
        
        let nextGuess: string;
        let waitTime: number;

        // Use selected agent to get next guess
        if (selectedAgent === "qwen-grpo") {
          // Calculate waiting time: normal distribution with mean=3s, variance=1 (seconds²)
          // Convert to milliseconds: mean=3000ms, variance=1s² = 1,000,000 ms²
          waitTime = Math.max(0, sampleNormal(3000, 1000000));
          waitTime = Math.min(waitTime, 10000); // Cap at 10 seconds max
          
          // Show thinking message and wait
          await waitWithThinking(waitTime, (msg) => {
            setMessage(`Qwen GRPO ${msg}`);
          });

          // Greedy solver approach
          nextGuess = greedySolver(guesses.length, remainingCandidates, gameHistory);
          
          console.log("=".repeat(80));
          console.log("QWEN GRPO - Greedy Solver Guess:");
          console.log(`Attempt ${guesses.length + 1}: ${nextGuess}`);
          console.log(`Remaining candidates: ${remainingCandidates.length}`);
          console.log(`Wait time: ${(waitTime / 1000).toFixed(2)}s`);
          console.log("=".repeat(80));
        } else {
          // Calculate waiting time: normal distribution with mean=7s, variance=2 (seconds²)
          // Convert to milliseconds: mean=7000ms, variance=2s² = 2,000,000 ms²
          waitTime = Math.max(0, sampleNormal(7000, 2000000));
          waitTime = Math.min(waitTime, 20000); // Cap at 20 seconds max
          
          // Show thinking message and wait
          await waitWithThinking(waitTime, (msg) => {
            setMessage(`Gemini ${msg}`);
          });

          // Gemini (Qwen LLM) approach
          const response = await fetch("/api/agent", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              history: gameHistory,
              attemptNumber: guesses.length + 1,
            }),
          });

          if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
          }

          const data = await response.json();
          nextGuess = data.guess?.trim() || "";
          const source = data.source || "unknown";
          const metadata = data.metadata || {};

          console.log("=".repeat(80));
          console.log("GEMINI - Qwen LLM Response:");
          console.log(`Attempt ${guesses.length + 1}: ${nextGuess}`);
          console.log(`Source: ${source.toUpperCase()}`);
          console.log(`Metadata:`, metadata);
          console.log(`Wait time: ${(waitTime / 1000).toFixed(2)}s`);
          console.log("=".repeat(80));

          if (!nextGuess || nextGuess.length !== EQUATION_LENGTH) {
            console.error("Invalid guess from Gemini:", nextGuess);
            throw new Error(`Invalid guess from Gemini: "${nextGuess}"`);
          }
        }

        // Validate the guess
        const validation = validateGuess(nextGuess);
        if (!validation.valid) {
          setMessage(`Agent made an invalid guess: ${validation.message}`);
          setIsAutoPlaying(false);
          return;
        }

        // Evaluate the guess
        const evaluation = getTileStates(nextGuess, solution);
        const nextGuesses = [...guesses, nextGuess];
        const nextFeedback = [...feedback, evaluation];
        const nextHistory = [...gameHistory, { guess: nextGuess, feedback: evaluation }];
        
        // For Qwen GRPO, filter remaining candidates
        if (selectedAgent === "qwen-grpo") {
          const filteredCandidates = filterCandidatesFromHistory(EQUATION_POOL, nextHistory);
          setRemainingCandidates(filteredCandidates);
          setMessage(`Qwen GRPO guessed: ${nextGuess} (${filteredCandidates.length} possibilities remaining)`);
        } else {
          setMessage(`Gemini guessed: ${nextGuess}`);
        }
        
        setGuesses(nextGuesses);
        setFeedback(nextFeedback);
        setGameHistory(nextHistory);

        if (nextGuess === solution) {
          setGameStatus("won");
          setMessage(`${selectedAgent === "qwen-grpo" ? "Qwen GRPO" : "Gemini"} solved it in ${nextGuesses.length} tries!`);
          setIsAutoPlaying(false);
          return;
        }

        if (nextGuesses.length >= MAX_GUESSES) {
          setGameStatus("lost");
          setMessage(`${selectedAgent === "qwen-grpo" ? "Qwen GRPO" : "Gemini"} ran out of guesses. Solution was ${solution}`);
          setIsAutoPlaying(false);
          return;
        }

        // Continue to next guess immediately (useEffect will trigger again)
        setGameStatus("playing");
        setMessage(null);

      } catch (error: any) {
        if (isCancelled) return;
        console.error("Error making guess:", error);
        setMessage(`Error: ${error.message}`);
        setIsAutoPlaying(false);
        setGameStatus("playing");
      }
    };

    // Start making guess
    makeGuess();

    return () => {
      isCancelled = true;
    };
  }, [isAutoPlaying, gameStatus, guesses.length, gameHistory, solution, feedback, remainingCandidates, EQUATION_POOL, selectedAgent]);

  const boardRows = useMemo(() => {
    const rows: DisplayTileState[][] = [];
    for (let i = 0; i < MAX_GUESSES; i++) {
      const row: DisplayTileState[] = [];
      for (let j = 0; j < EQUATION_LENGTH; j++) {
        if (i < guesses.length) {
          row.push(feedback[i][j]);
        } else if (i === guesses.length && gameStatus === "loading") {
          row.push("pending");
        } else {
          row.push("empty");
        }
      }
      rows.push(row);
    }
    return rows;
  }, [guesses, feedback, gameStatus]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen w-full max-w-4xl flex-col items-center px-4 py-10 sm:px-6">
        <header className="mb-8 text-center">
          <div className="flex items-center justify-between w-full mb-4">
            <Link
              href="/"
              className="text-sm text-slate-400 hover:text-slate-200 underline"
            >
              ← Back to Player Mode
            </Link>
          </div>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">Nerdle Agent Mode</h1>
          <p className="mt-2 text-sm text-slate-300 sm:text-base">
            Compare different agents playing Nerdle autonomously
          </p>
        </header>

        <section className="flex w-full flex-1 flex-col items-center gap-8">
          {/* Agent Selector */}
          <div className="w-full rounded-md border border-slate-800 bg-slate-900/80 px-4 py-3">
            <div className="mb-3">
              <p className="text-xs text-slate-400 mb-2">Select Agent</p>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedAgent("qwen-grpo")}
                  disabled={isAutoPlaying}
                  className={`px-4 py-2 rounded-md text-sm font-semibold transition ${
                    selectedAgent === "qwen-grpo"
                      ? "bg-emerald-500 text-emerald-950"
                      : "bg-slate-700 text-slate-200 hover:bg-slate-600"
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  Qwen GRPO
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedAgent("gemini")}
                  disabled={isAutoPlaying}
                  className={`px-4 py-2 rounded-md text-sm font-semibold transition ${
                    selectedAgent === "gemini"
                      ? "bg-emerald-500 text-emerald-950"
                      : "bg-slate-700 text-slate-200 hover:bg-slate-600"
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  Gemini
                </button>
              </div>
            </div>
          </div>

          {/* Solver Controls */}
          <div className="w-full rounded-md border border-slate-800 bg-slate-900/80 px-4 py-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-400">Agent Status</p>
                <p className="text-sm font-semibold text-slate-100">
                  {selectedAgent === "qwen-grpo" ? "Qwen GRPO" : "Gemini"}
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => {
                    if (gameStatus === "won" || gameStatus === "lost") {
                      resetGame();
                    }
                    setIsAutoPlaying(!isAutoPlaying);
                  }}
                  disabled={gameStatus === "loading"}
                  className="rounded-md bg-emerald-500 px-4 py-2 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-400 disabled:opacity-50"
                >
                  {isAutoPlaying ? "Pause" : gameStatus === "playing" ? "Start Auto-Play" : "New Game"}
                </button>
              </div>
            </div>
          </div>

          {/* Game Board */}
          <div className="flex w-full flex-col items-center gap-2">
            {boardRows.map((row, rowIndex) => (
              <div key={rowIndex} className="grid w-full grid-cols-8 gap-2">
                {row.map((state, colIndex) => (
                  <span
                    key={colIndex}
                    className={tileClassName(state)}
                  >
                    {rowIndex < guesses.length ? guesses[rowIndex][colIndex] : ""}
                  </span>
                ))}
              </div>
            ))}
          </div>

          {/* Status Message */}
          {message && (
            <div className="w-full rounded-md border border-slate-800 bg-slate-900/80 px-4 py-3 text-center text-sm text-slate-300">
              {message}
            </div>
          )}

          {/* Solver Info (only for Qwen GRPO) */}
          {selectedAgent === "qwen-grpo" && remainingCandidates.length > 0 && remainingCandidates.length < EQUATION_POOL.length && (
            <div className="w-full rounded-md border border-slate-800 bg-slate-900/80 px-4 py-3">
              <p className="text-xs text-slate-400 mb-2">Qwen GRPO Status:</p>
              <p className="text-sm text-slate-300">
                {remainingCandidates.length} possible solution{remainingCandidates.length !== 1 ? 's' : ''} remaining
              </p>
            </div>
          )}

          {/* Game Info */}
          <footer className="w-full rounded-md border border-slate-800 bg-slate-900/80 px-4 py-3 text-xs text-slate-300 sm:text-sm">
            <p>
              Attempts: {guesses.length} / {MAX_GUESSES}
              {gameStatus === "won" && ` - Solved in ${guesses.length} tries!`}
              {gameStatus === "lost" && ` - Solution was ${solution}`}
            </p>
          </footer>
        </section>
      </div>
    </div>
  );
}

/**
 * Sample from a normal distribution using Box-Muller transform
 * @param mean Mean of the distribution
 * @param variance Variance of the distribution (std dev = sqrt(variance))
 * @returns A random number from normal distribution
 */
function sampleNormal(mean: number, variance: number): number {
  const stdDev = Math.sqrt(variance);
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z0 * stdDev;
}

/**
 * Wait for a specified time and show thinking message
 * @param milliseconds Time to wait in milliseconds
 * @param onThinking Callback to show thinking message
 */
function waitWithThinking(
  milliseconds: number,
  onThinking: (message: string) => void
): Promise<void> {
  return new Promise((resolve) => {
    onThinking("thinking...");
    setTimeout(() => {
      resolve();
    }, milliseconds);
  });
}

/**
 * Greedy solver for Nerdle
 * - First guess: "48-32=16" (optimal starting guess)
 * - Subsequent guesses: randomly pick from remaining valid candidates
 */
function greedySolver(
  attemptNumber: number,
  remainingCandidates: string[],
  history: GameHistory[]
): string {
  // First guess is always the optimal starting guess
  if (attemptNumber === 0) {
    return "48-32=16";
  }

  // If no remaining candidates (shouldn't happen), return fallback
  if (remainingCandidates.length === 0) {
    console.warn("No remaining candidates, using fallback");
    return "10+20=30";
  }

  // Randomly pick from remaining candidates
  const randomIndex = Math.floor(Math.random() * remainingCandidates.length);
  return remainingCandidates[randomIndex];
}

/**
 * Filter candidate equations based on feedback from all previous guesses
 * Only keeps equations that would produce the same feedback for all guesses
 */
function filterCandidatesFromHistory(
  candidates: string[],
  history: GameHistory[]
): string[] {
  return candidates.filter((candidate) => {
    // Check if this candidate would produce the same feedback for all guesses
    for (const historyItem of history) {
      const expectedFeedback = historyItem.feedback;
      const candidateFeedback = getTileStates(historyItem.guess, candidate);
      
      // If feedback doesn't match, this candidate is invalid
      if (!feedbackArraysMatch(expectedFeedback, candidateFeedback)) {
        return false;
      }
    }
    // All feedback matches
    return true;
  });
}

/**
 * Check if two feedback arrays match exactly
 */
function feedbackArraysMatch(feedback1: TileState[], feedback2: TileState[]): boolean {
  if (feedback1.length !== feedback2.length) {
    return false;
  }
  
  for (let i = 0; i < feedback1.length; i++) {
    if (feedback1[i] !== feedback2[i]) {
      return false;
    }
  }
  
  return true;
}

function pickRandomEquation(pool: string[]): string {
  if (pool.length === 0) {
    return "10+32=42";
  }
  const index = Math.floor(Math.random() * pool.length);
  return pool[index];
}

function generateEquationPool(): string[] {
  const results = new Set<string>();
  const operations: Array<{ symbol: string; calculate: (a: number, b: number) => number | null }> = [
    { symbol: "+", calculate: (a, b) => a + b },
    { symbol: "-", calculate: (a, b) => a - b },
    { symbol: "*", calculate: (a, b) => a * b },
    {
      symbol: "/",
      calculate: (a, b) => {
        if (b === 0 || a % b !== 0) {
          return null;
        }
        return a / b;
      },
    },
  ];

  const noLeadingZero = (value: number) => {
    const text = String(value);
    return text.length === 1 || !text.startsWith("0");
  };

  for (let left = 0; left < 100; left += 1) {
    if (!noLeadingZero(left)) {
      continue;
    }
    for (let right = 0; right < 100; right += 1) {
      if (!noLeadingZero(right)) {
        continue;
      }
      for (const op of operations) {
        const result = op.calculate(left, right);
        if (result === null || result < 0 || !Number.isInteger(result)) {
          continue;
        }
        if (!noLeadingZero(result)) {
          continue;
        }
        const equation = `${left}${op.symbol}${right}=${result}`;
        if (equation.length === EQUATION_LENGTH) {
          results.add(equation);
        }
      }
    }
  }

  return Array.from(results);
}

function validateGuess(guess: string): { valid: true } | { valid: false; message: string } {
  if (guess.length !== EQUATION_LENGTH) {
    return { valid: false, message: `Guesses must be ${EQUATION_LENGTH} characters.` };
  }

  if (![...guess].every((char) => ALLOWED_INPUT.has(char))) {
    return { valid: false, message: "Use only digits and the symbols + - * / =." };
  }

  if (guess.split("=").length !== 2) {
    return { valid: false, message: "Equation must contain exactly one equals sign." };
  }

  const [left, right] = guess.split("=") as [string, string];
  if (!left || !right) {
    return { valid: false, message: "Equation has to include values on both sides of =." };
  }

  const leftValue = safeEvaluate(left);
  const rightValue = safeEvaluate(right);

  if (leftValue === null || rightValue === null) {
    return { valid: false, message: "Equation could not be evaluated." };
  }

  if (!Number.isInteger(leftValue) || !Number.isInteger(rightValue)) {
    return { valid: false, message: "Stick to whole-number results." };
  }

  if (leftValue !== rightValue) {
    return { valid: false, message: "Equation does not balance. Try again!" };
  }

  return { valid: true };
}

function safeEvaluate(expression: string): number | null {
  try {
    const result = Function(`"use strict"; return (${expression})`)();
    return typeof result === "number" && !isNaN(result) ? result : null;
  } catch {
    return null;
  }
}

function getTileStates(guess: string, target: string): TileState[] {
  const result: TileState[] = Array(EQUATION_LENGTH).fill("absent");
  const targetChars = target.split("");
  const used = Array(EQUATION_LENGTH).fill(false);

  for (let i = 0; i < EQUATION_LENGTH; i += 1) {
    if (guess[i] === target[i]) {
      result[i] = "correct";
      used[i] = true;
    }
  }

  for (let i = 0; i < EQUATION_LENGTH; i += 1) {
    if (result[i] === "correct") {
      continue;
    }
    const guessChar = guess[i];
    const matchIndex = targetChars.findIndex((char, idx) => !used[idx] && char === guessChar);
    if (matchIndex !== -1) {
      result[i] = "present";
      used[matchIndex] = true;
    }
  }

  return result;
}

function tileClassName(state: DisplayTileState): string {
  const base =
    "flex h-14 w-10 items-center justify-center rounded-md border-2 text-xl font-bold uppercase sm:h-16 sm:w-14 sm:text-2xl";

  switch (state) {
    case "correct":
      return `${base} border-emerald-500 bg-emerald-500 text-slate-950`;
    case "present":
      return `${base} border-purple-500 bg-purple-500 text-slate-950`;
    case "absent":
      return `${base} border-slate-700 bg-slate-900 text-slate-500`;
    case "pending":
      return `${base} border-slate-600 bg-slate-800 text-slate-400 animate-pulse`;
    default:
      return `${base} border-slate-700 bg-slate-900 text-slate-500`;
  }
}

