using System.Net.Http.Headers;

namespace Connect4;

sealed class Connect4Board
{
    readonly int[,] board = new int[7,6];
    readonly int[] heights = new int[7];
    readonly Stack<int> history = new();

    public int Player { get; private set; } = 1;
    public int Opponent { get; private set; } = 2;

    public int Height(int column) => heights[column];
    public int this[int column, int row] => 
        column >= 0 && column < 7 &&
        row >= 0 && row < heights[column]
        ? board[column, row] : 0;

    public int Winner { get; private set; }
    public bool Full { get; private set; }
    public bool Finished => Winner != 0 || Full;
    public bool CanUndo => history.Count > 0;

    public int[] ValidMoves => Enumerable.Range(0, 7).Where(column => heights[column] < 6).ToArray();

    public void Move(int column)
    {
        if (Finished) throw new InvalidOperationException("The game already finished.");
        if (column < 0 || column > 6) throw new ArgumentOutOfRangeException(paramName: nameof(column), actualValue: column, message: "The move index must be greater or equal to zero and less than 7.");
        var height = heights[column];
        if (height > 5) throw new InvalidOperationException("This column is already full.");

        board[column, height] = Player;
        heights[column] = height + 1;

        Full = heights.All(h => h > 5);

        try 
        {
            //
            // horizontal line
            //
            int count = 1;
            if (this[column-1, height] == Player)
            {
                count += 1;
                if (this[column-2, height] == Player)
                {
                    count +=1;
                    if (this[column-3, height] == Player)
                    {
                        Winner = Player;
                        return;
                    }
                }    
            }
            if (this[column+1, height] == Player)
            {
                count +=1; if (count == 4) { Winner = Player; return; }
                if (this[column+2, height] == Player)
                {
                    count +=1; if (count == 4) { Winner = Player; return; }
                    if (this[column+3, height] == Player)
                    {
                        Winner = Player; return;
                    }                
                }
            }

            //
            // vertical line
            //
            count = 1;
            if (this[column, height+1] == Player)
            {
                count += 1;
                if (this[column, height+2] == Player)
                {
                    count +=1;
                    if (this[column, height + 3] == Player)
                    {
                        Winner = Player;
                        return;
                    }
                }
            }
            if (this[column, height - 1] == Player)
            {
                count +=1; if (count == 4) { Winner = Player; return; }
                if (this[column, height-2] == Player)
                {
                    count +=1; if (count == 4) { Winner = Player; return; }
                    if (this[column, height - 3] == Player)
                    {
                        Winner = Player; return;
                    }
                }
            }

            //
            // upper left to lower right line
            //
            count = 1;
            if (this[column-1, height+1] == Player)
            {
                count += 1;
                if (this[column-2, height+2] == Player)
                {
                    count +=1;
                    if (this[column - 3, height + 3] == Player)
                    {
                        Winner = Player;
                        return;
                    }
                }
            }
            if (this[column+1, height - 1] == Player)
            {
                count +=1; if (count == 4) { Winner = Player; return; }
                if (this[column+2, height-2] == Player)
                {
                    count +=1; if (count == 4) { Winner = Player; return; }
                    if (this[column+3, height - 3] == Player)
                    {
                        Winner = Player; 
                        return;
                    }
                }
            }

            //
            // upper right to lower left line
            //
            count = 1;
            if (this[column+1, height+1] == Player)
            {
                count += 1;
                if (this[column + 2, height+2] == Player)
                {
                    count +=1;
                    if (this[column + 3, height + 3] == Player)
                    {
                        Winner = Player;
                        return;
                    }
                }
            }
            if (this[column-1, height - 1] == Player)
            {
                count +=1; if (count == 4) { Winner = Player; return; }
                if (this[column-2, height-2] == Player)
                {
                    count +=1; if (count == 4) { Winner = Player; return; }
                    if (this[column-3, height - 3] == Player)
                    {
                        Winner = Player; 
                        return;
                    }
                }
            }
        }
        finally
        {
            (Player, Opponent) = (Opponent, Player);
        }
    }
    public int Undo() 
    {
        if (history.Count == 0) return -1;
        var column = history.Pop();
        heights[column] -= 1;
        Winner = 0; Full = false;
        (Player, Opponent) = (Opponent, Player);
        return column;
    }
    public Connect4Board Clone()
    {
        var copy = new Connect4Board
        {
            Player = Player,
            Opponent = Opponent
        };
        Array.Copy(board, copy.board, board.Length);
        Array.Copy(heights, copy.heights, heights.Length);
        foreach (var column in history.Reverse()) copy.history.Push(column);
        copy.Winner = Winner;
        copy.Full = Full;
        return copy;
    }
}
